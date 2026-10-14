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

import os
from dataclasses import dataclass, field

from aiohomekit.model.characteristics.const import (
    AudioCodecValues,
    IPVersionValues,
    SampleRateValues,
    SessionControlCommandValues,
    SetupEndpointStatusValues,
    SRTPCryptoSuiteValues,
    StreamingStatusValues,
)
from aiohomekit.model.characteristics.structs import (
    AudioCodecParameters,
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


@dataclass
class StreamSession:
    """A negotiated streaming session with a paired camera."""

    session_id: bytes
    accessory_address: RTPEndpointAddress
    video_ssrc: int
    audio_ssrc: int
    controller_video_srtp: SRTPParameters
    controller_audio_srtp: SRTPParameters
    selected: SelectedRTPStreamConfiguration = field(repr=False)


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
        import base64

        return struct_cls.decode(base64.b64decode(value))

    async def _write_struct(self, name: str, struct) -> None:
        import base64

        iid = self.iids[name]
        value = base64.b64encode(struct.encode()).decode("ascii")
        await self.pairing.put_characteristics([(self.aid, iid, value)])

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
            session_id, video_cfg, audio_cfg, video_srtp, audio_srtp, response, width, height, fps
        )
        await self._write_struct("selected_rtp", selected)

        return StreamSession(
            session_id=session_id,
            accessory_address=response.accessory_address,
            video_ssrc=response.video_ssrc,
            audio_ssrc=response.audio_ssrc,
            controller_video_srtp=video_srtp,
            controller_audio_srtp=audio_srtp,
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
        attrs = vconf.video_attrs[0]
        for candidate in vconf.video_attrs:
            if candidate.width == width and candidate.height == height:
                attrs = candidate
                break
        if width and height:
            attrs = type(attrs)(width=width, height=height, fps=fps or attrs.fps)

        video_params = SelectedVideoParameters(
            codec_type=vconf.codec_type,
            codec_parameters=vconf.codec_params[0],
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
            codec = aconf.codec
            params = aconf.parameters[0] if aconf.parameters else AudioCodecParameters(
                audio_channels=1, bit_rate=0, sample_rate=SampleRateValues.TWENTY_FOUR_KHZ, rtp_time=20
            )
            audio_params = SelectedAudioParameters(
                codec_type=codec if codec is not None else AudioCodecValues.OPUS,
                codec_parameters=params,
                rtp_params=None,
                comfort_noise=0,
            )

        return SelectedRTPStreamConfiguration(
            control=SessionControl(
                session=session_id, command=SessionControlCommandValues.START_SESSION
            ),
            video_params=video_params,
            audio_params=audio_params,
        )
