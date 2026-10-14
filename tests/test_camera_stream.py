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
                              width=1280, height=720, fps=30, max_bitrate=2000)

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
    assert sel.video_params.rtp_params.max_bitrate == 2000
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


@pytest.mark.asyncio
async def test_sdp_uses_accessory_keys_and_relay_ports():
    """The receive SDP decrypts with the accessory's response keys, never ours."""
    pairing = FakePairing()
    pairing.values[IIDS["supported_audio"]] = b64(SupportedAudioStreamConfiguration(config=[AudioCodecConfiguration(
        codec=AudioCodecValues.AAC_ELD,
        parameters=[AudioCodecParameters(audio_channels=1, bit_rate=BitRateValues.VARIABLE,
                                         sample_rate=SampleRateValues.SIXTEEN_KHZ, rtp_time=30)])],
        comfort_noise=0))
    session = await CameraStream(pairing, AID, IIDS).start(local_ip="192.0.2.35", video_port=50000, audio_port=50002)

    sdp = session.sdp("127.0.0.1", video_port=41000, audio_port=41002)

    assert "c=IN IP4 127.0.0.1" in sdp
    assert "m=video 41000 RTP/SAVP 99" in sdp
    assert "m=audio 41002 RTP/SAVP 110" in sdp
    assert "a=rtpmap:110 mpeg4-generic/16000/1" in sdp
    assert "config=F8F03000" in sdp  # what fdk-aac reports for 16 kHz mono ELD with 480-sample frames
    acc = accessory_response()
    for srtp in (acc.video_srtp, acc.audio_srtp):
        assert base64.b64encode(srtp.master_key + srtp.master_salt).decode() in sdp
    for srtp in (session.controller_video_srtp, session.controller_audio_srtp):
        assert base64.b64encode(srtp.master_key + srtp.master_salt).decode() not in sdp


@pytest.mark.asyncio
async def test_sdp_defaults_to_offered_ports_and_skips_non_eld_audio():
    session = await CameraStream(FakePairing(), AID, IIDS).start(local_ip="192.0.2.35", video_port=50000, audio_port=50002)
    sdp = session.sdp("192.0.2.35")
    assert "m=video 50000 RTP/SAVP 99" in sdp
    assert "m=audio" not in sdp  # the default fixture negotiates Opus
