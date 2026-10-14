#
# Copyright 2026 aiohomekit team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""Live camera streaming for a paired HomeKit camera or doorbell.

The controller side of the HAP streaming protocol: read what the camera
supports, offer our receive endpoints over Setup Endpoints, then start and
stop the RTP session through Selected RTP Stream Configuration.

This module owns the negotiation and session lifecycle only. Moving the SRTP
media packets is layered on top of the session it returns.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field

from aiohomekit.model.characteristics.const import (
    AudioCodecValues,
    CVOEnabledValues,
    PacketizationModeValues,
    ProfileIDValues,
    ProfileSupportLevelValues,
    IPVersionValues,
    SampleRateValues,
    SessionControlCommandValues,
    SetupEndpointStatusValues,
    SRTPCryptoSuiteValues,
    StreamingStatusValues,
    VideoCodecTypeValues,
)
from aiohomekit.model.characteristics.structs import (
    AudioCodecParameters,
    AudioRTPParameters,
    VideoAttrs,
    VideoCodecParameters,
    RTPEndpointAddress,
    SelectedAudioParameters,
    SelectedRTPStreamConfiguration,
    SelectedVideoParameters,
    SessionControl,
    SetupEndpointsRequest,
    SetupEndpointsResponse,
    SRTPParameters,
    StreamingStatus,
    SupportedAudioStreamConfiguration,
    SupportedVideoStreamConfiguration,
    VideoRTPParameters,
)

# A HAP streaming service groups these characteristics; callers map each name
# to the (aid, iid) of the paired camera.
STREAM_CHARACTERISTICS = (
    "streaming_status",
    "supported_video",
    "supported_audio",
    "supported_rtp",
    "setup_endpoints",
    "selected_rtp",
)

SRTP_KEY_LEN = 16
SRTP_SALT_LEN = 14


def _new_srtp() -> SRTPParameters:
    """Fresh SRTP keys for one direction of a stream."""
    return SRTPParameters(
        crypto_suite=SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80,
        master_key=os.urandom(SRTP_KEY_LEN),
        master_salt=os.urandom(SRTP_SALT_LEN),
    )


# libsrtp / ffmpeg name for each HAP crypto suite.
_SRTP_SUITE_NAME = {
    SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80: "AES_CM_128_HMAC_SHA1_80",
    SRTPCryptoSuiteValues.AES_256_CM_HMAC_SHA1_80: "AES_256_CM_HMAC_SHA1_80",
}


@dataclass
class StreamSession:
    """A negotiated streaming session with a paired camera.

    Each endpoint encrypts with its own keys and announces them during Setup
    Endpoints: the accessory's downstream media uses the keys it returned in
    the response, so those are what decrypts the video and audio we receive.
    The keys we sent in the request are for the talk-back we encrypt to it.
    """

    session_id: bytes
    accessory_address: RTPEndpointAddress
    local_video_port: int
    local_audio_port: int
    video_ssrc: int
    audio_ssrc: int
    controller_video_srtp: SRTPParameters
    controller_audio_srtp: SRTPParameters
    accessory_video_srtp: SRTPParameters = field(repr=False)
    accessory_audio_srtp: SRTPParameters = field(repr=False)
    selected: SelectedRTPStreamConfiguration = field(repr=False)

    def video_sdp(self, local_ip: str) -> str:
        """SDP describing the accessory's H264 video for an SRTP consumer.

        Feed it to ffmpeg (-protocol_whitelist file,crypto,data,rtp,udp,srtp)
        and libsrtp decrypts with the accessory's keys named in the response.
        """
        srtp = self.accessory_video_srtp
        suite = _SRTP_SUITE_NAME[srtp.crypto_suite]
        inline = base64.b64encode(srtp.master_key + srtp.master_salt).decode("ascii")
        return (
            "v=0\r\n"
            f"o=- 0 0 IN IP4 {local_ip}\r\n"
            "s=HomeKit\r\n"
            f"c=IN IP4 {local_ip}\r\n"
            "t=0 0\r\n"
            f"m=video {self.local_video_port} RTP/SAVP 99\r\n"
            "a=rtpmap:99 H264/90000\r\n"
            "a=fmtp:99 packetization-mode=1\r\n"
            f"a=crypto:1 {suite} inline:{inline}\r\n"
        )


class CameraStream:
    """Negotiates and controls one camera's RTP stream over an existing pairing."""

    def __init__(self, pairing, aid: int, iids: dict[str, int]) -> None:
        missing = [n for n in STREAM_CHARACTERISTICS if n not in iids]
        if missing:
            raise ValueError(f"missing stream characteristic iids: {missing}")
        self.pairing = pairing
        self.aid = aid
        self.iids = iids

    async def _read_struct(self, name: str, struct_cls):
        iid = self.iids[name]
        result = await self.pairing.get_characteristics([(self.aid, iid)])
        value = result[(self.aid, iid)].get("value")
        if value is None:
            return None

        return struct_cls.decode(base64.b64decode(value))

    async def _write_struct(self, name: str, struct) -> None:
        iid = self.iids[name]
        value = base64.b64encode(struct.encode()).decode("ascii")
        result = await self.pairing.put_characteristics([(self.aid, iid, value)])
        if result:
            raise RuntimeError(f"write to {name} (iid {iid}) failed: {result}")

    async def streaming_status(self) -> StreamingStatusValues | None:
        status = await self._read_struct("streaming_status", StreamingStatus)
        return status.status if status else None

    async def supported_video(self) -> SupportedVideoStreamConfiguration | None:
        return await self._read_struct("supported_video", SupportedVideoStreamConfiguration)

    async def supported_audio(self) -> SupportedAudioStreamConfiguration | None:
        return await self._read_struct("supported_audio", SupportedAudioStreamConfiguration)

    async def start(
        self,
        *,
        local_ip: str,
        video_port: int,
        audio_port: int,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        audio: bool = True,
    ) -> StreamSession:
        """Offer our endpoints, then ask the camera to start sending."""
        video_cfg = await self.supported_video()
        audio_cfg = await self.supported_audio()
        if not video_cfg or not video_cfg.config:
            raise RuntimeError("camera advertises no supported video configuration")

        session_id = os.urandom(16)
        video_srtp, audio_srtp = _new_srtp(), _new_srtp()

        request = SetupEndpointsRequest(
            session_id=session_id,
            controller_address=RTPEndpointAddress(
                ip_version=IPVersionValues.IPV6 if ":" in local_ip else IPVersionValues.IPV4,
                ip_address=local_ip,
                video_rtp_port=video_port,
                audio_rtp_port=audio_port,
            ),
            video_srtp=video_srtp,
            audio_srtp=audio_srtp,
        )
        await self._write_struct("setup_endpoints", request)

        response = await self._read_struct("setup_endpoints", SetupEndpointsResponse)
        if response is None:
            raise RuntimeError("no Setup Endpoints response from camera")
        if response.status != SetupEndpointStatusValues.SUCCESS:
            raise RuntimeError(f"camera refused Setup Endpoints: {response.status!r}")

        selected = self._build_selected(
            session_id, video_cfg, audio_cfg if audio else None, video_srtp, audio_srtp, response, width, height, fps
        )
        await self._write_struct("selected_rtp", selected)

        return StreamSession(
            session_id=session_id,
            accessory_address=response.accessory_address,
            local_video_port=video_port,
            local_audio_port=audio_port,
            video_ssrc=response.video_ssrc,
            audio_ssrc=response.audio_ssrc,
            controller_video_srtp=video_srtp,
            controller_audio_srtp=audio_srtp,
            accessory_video_srtp=response.video_srtp,
            accessory_audio_srtp=response.audio_srtp,
            selected=selected,
        )

    async def stop(self, session: StreamSession) -> None:
        """Tell the camera to end the session."""
        end = SelectedRTPStreamConfiguration(
            control=SessionControl(
                session=session.session_id, command=SessionControlCommandValues.END_SESSION
            ),
            video_params=None,
            audio_params=None,
        )
        await self._write_struct("selected_rtp", end)

    def _build_selected(
        self, session_id, video_cfg, audio_cfg, video_srtp, audio_srtp, response, width, height, fps
    ) -> SelectedRTPStreamConfiguration:
        vconf = video_cfg.config[0]
        # The camera may split resolutions across several config entries; gather them all.
        all_attrs = [a for cfg in video_cfg.config for a in (cfg.video_attrs or [])]
        attrs = next((a for a in all_attrs if a.width == width and a.height == height), None)
        if attrs is None:
            attrs = all_attrs[0] if all_attrs else VideoAttrs(width=1280, height=720, fps=30)
        if width and height:
            attrs = VideoAttrs(width=width, height=height, fps=fps or attrs.fps)

        # Pick a profile/level the camera listed, but always send a fully specified
        # codec parameter set (profile, level, packetization) or the accessory rejects it.
        listed = [cp for cfg in video_cfg.config for cp in (cfg.codec_params or [])]
        profile = next((cp.profile_id for cp in listed if cp.profile_id is not None), ProfileIDValues.MAIN_PROFILE)
        level = next((cp.level for cp in listed if cp.level is not None), ProfileSupportLevelValues.THREE_ONE)
        codec_params = VideoCodecParameters(
            profile_id=profile,
            level=level,
            packetization_mode=PacketizationModeValues.NON_INTERLEAVED_MODE,
            cvo_enabled=CVOEnabledValues.NOT_SUPPORTED,
            cvo_id=None,
        )

        video_params = SelectedVideoParameters(
            codec_type=vconf.codec_type if vconf.codec_type is not None else VideoCodecTypeValues.H264,
            codec_parameters=codec_params,
            video_attrs=attrs,
            rtp_params=VideoRTPParameters(
                payload_type=99,
                ssrc=response.video_ssrc or int.from_bytes(os.urandom(4), "little"),
                max_bitrate=800,
                min_rtcp_interval=0.5,
                max_mtu=1378,
            ),
        )

        audio_params = None
        if audio_cfg and audio_cfg.config:
            aconf = audio_cfg.config[0]
            src = aconf.parameters[0] if aconf.parameters else None
            params = AudioCodecParameters(
                audio_channels=src.audio_channels if src and src.audio_channels else 1,
                bit_rate=src.bit_rate if src and src.bit_rate is not None else 0,
                sample_rate=src.sample_rate if src and src.sample_rate is not None else SampleRateValues.SIXTEEN_KHZ,
                rtp_time=src.rtp_time if src and src.rtp_time else 30,
            )
            audio_params = SelectedAudioParameters(
                codec_type=aconf.codec if aconf.codec is not None else AudioCodecValues.AAC_ELD,
                codec_parameters=params,
                rtp_params=AudioRTPParameters(
                    payload_type=110,
                    ssrc=int.from_bytes(os.urandom(4), "little"),
                    max_bitrate=24,
                    min_rtcp_interval=5.0,
                    comfort_noise_payload_type=13,
                ),
                comfort_noise=0,
            )

        return SelectedRTPStreamConfiguration(
            control=SessionControl(
                session=session_id, command=SessionControlCommandValues.START_SESSION
            ),
            video_params=video_params,
            audio_params=audio_params,
        )
