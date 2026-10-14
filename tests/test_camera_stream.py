import base64
import pytest

from aiohomekit.camera import CameraStream, STREAM_CHARACTERISTICS
from aiohomekit.model.characteristics.const import (
    AudioCodecValues, BitRateValues, CVOEnabledValues, IPVersionValues,
    PacketizationModeValues, ProfileIDValues, ProfileSupportLevelValues,
    SampleRateValues, SessionControlCommandValues, SetupEndpointStatusValues,
    SRTPCryptoSuiteValues, VideoCodecTypeValues,
)
from aiohomekit.model.characteristics.structs import (
    AudioCodecConfiguration, AudioCodecParameters, RTPEndpointAddress,
    SelectedRTPStreamConfiguration, SetupEndpointsRequest, SetupEndpointsResponse,
    SRTPParameters, SupportedAudioStreamConfiguration, SupportedVideoStreamConfiguration,
    VideoAttrs, VideoCodecParameters, VideoConfigConfiguration,
)

IIDS = {name: i + 10 for i, name in enumerate(STREAM_CHARACTERISTICS)}
AID = 1


def b64(struct):
    return base64.b64encode(struct.encode()).decode()


def supported_video():
    return SupportedVideoStreamConfiguration(config=[VideoConfigConfiguration(
        codec_type=VideoCodecTypeValues.H264,
        codec_params=[VideoCodecParameters(
            profile_id=ProfileIDValues.MAIN_PROFILE, level=ProfileSupportLevelValues.FOUR,
            packetization_mode=PacketizationModeValues.NON_INTERLEAVED_MODE,
            cvo_enabled=CVOEnabledValues.NOT_SUPPORTED, cvo_id=0)],
        video_attrs=[VideoAttrs(width=1920, height=1080, fps=30),
                     VideoAttrs(width=1280, height=720, fps=30)])])


def supported_audio():
    return SupportedAudioStreamConfiguration(config=[AudioCodecConfiguration(
        codec=AudioCodecValues.OPUS,
        parameters=[AudioCodecParameters(audio_channels=1, bit_rate=BitRateValues.VARIABLE,
                                         sample_rate=SampleRateValues.TWENTY_FOUR_KHZ, rtp_time=20)])],
        comfort_noise=0)


def accessory_response():
    return SetupEndpointsResponse(
        session_id=b"\x00" * 16, status=SetupEndpointStatusValues.SUCCESS,
        accessory_address=RTPEndpointAddress(ip_version=IPVersionValues.IPV4,
            ip_address="192.0.2.26", video_rtp_port=60000, audio_rtp_port=60002),
        video_srtp=SRTPParameters(SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80, b"\xaa"*16, b"\xbb"*14),
        audio_srtp=SRTPParameters(SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80, b"\xcc"*16, b"\xdd"*14),
        video_ssrc=0x11223344, audio_ssrc=0x55667788)


class FakePairing:
    """Serves canned supported configs; records writes; echoes Setup Endpoints."""

    def __init__(self):
        self.writes = []
        self.setup_echo = accessory_response()
        self.values = {
            IIDS["supported_video"]: b64(supported_video()),
            IIDS["supported_audio"]: b64(supported_audio()),
            IIDS["setup_endpoints"]: b64(self.setup_echo),
        }

    async def get_characteristics(self, chars, **kw):
        out = {}
        for aid, iid in chars:
            out[(aid, iid)] = {"value": self.values.get(iid)}
        return out

    async def put_characteristics(self, chars):
        for aid, iid, value in chars:
            self.writes.append((iid, value))
            if iid == IIDS["setup_endpoints"]:
                self.values[iid] = b64(self.setup_echo)  # device answers with its endpoints
        return {}


@pytest.mark.asyncio
async def test_start_negotiates_and_starts():
    pairing = FakePairing()
    cam = CameraStream(pairing, AID, IIDS)
    session = await cam.start(local_ip="192.0.2.35", video_port=50000, audio_port=50002,
                              width=1280, height=720, fps=30)

    assert session.accessory_address.ip_address == "192.0.2.26"
    assert session.video_ssrc == 0x11223344
    assert len(session.controller_video_srtp.master_key) == 16

    writes = dict(pairing.writes)
    req = SetupEndpointsRequest.decode(base64.b64decode(writes[IIDS["setup_endpoints"]]))
    assert req.controller_address.ip_address == "192.0.2.35"
    assert req.controller_address.video_rtp_port == 50000

    sel = SelectedRTPStreamConfiguration.decode(base64.b64decode(writes[IIDS["selected_rtp"]]))
    assert sel.control.command == SessionControlCommandValues.START_SESSION
    assert sel.video_params.video_attrs.width == 1280
    assert sel.video_params.video_attrs.height == 720
    assert sel.audio_params is not None


@pytest.mark.asyncio
async def test_stop_ends_session():
    pairing = FakePairing()
    cam = CameraStream(pairing, AID, IIDS)
    session = await cam.start(local_ip="192.0.2.35", video_port=50000, audio_port=50002)
    pairing.writes.clear()
    await cam.stop(session)
    iid, value = pairing.writes[-1]
    sel = SelectedRTPStreamConfiguration.decode(base64.b64decode(value))
    assert iid == IIDS["selected_rtp"]
    assert sel.control.command == SessionControlCommandValues.END_SESSION


def test_missing_iids_rejected():
    with pytest.raises(ValueError):
        CameraStream(FakePairing(), AID, {"streaming_status": 1})


def test_stream_session_video_sdp():
    """video_sdp uses the accessory's response keys and our local port."""
    import base64
    from aiohomekit.camera import StreamSession
    from aiohomekit.model.characteristics.const import SRTPCryptoSuiteValues
    from aiohomekit.model.characteristics.structs import (
        RTPEndpointAddress,
        SelectedRTPStreamConfiguration,
        SRTPParameters,
    )

    acc = SRTPParameters(
        crypto_suite=SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80,
        master_key=bytes(range(16)),
        master_salt=bytes(range(14)),
    )
    ctrl = SRTPParameters(
        crypto_suite=SRTPCryptoSuiteValues.AES_CM_128_HMAC_SHA1_80,
        master_key=bytes(range(16, 32)),
        master_salt=bytes(range(14, 28)),
    )
    session = StreamSession(
        session_id=b"\x00" * 16,
        accessory_address=RTPEndpointAddress(
            ip_version=0, ip_address="192.0.2.26", video_rtp_port=5000, audio_rtp_port=5002
        ),
        local_video_port=63314,
        local_audio_port=63316,
        video_ssrc=1,
        audio_ssrc=2,
        controller_video_srtp=ctrl,
        controller_audio_srtp=ctrl,
        accessory_video_srtp=acc,
        accessory_audio_srtp=acc,
        selected=SelectedRTPStreamConfiguration(control=None, video_params=None, audio_params=None),
    )
    sdp = session.video_sdp("192.0.2.104")
    assert "m=video 63314 RTP/SAVP 99" in sdp
    assert "c=IN IP4 192.0.2.104" in sdp
    inline = base64.b64encode(acc.master_key + acc.master_salt).decode()
    assert f"a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:{inline}" in sdp
    # the controller keys are for talk-back, never the receive SDP
    assert base64.b64encode(ctrl.master_key + ctrl.master_salt).decode() not in sdp
