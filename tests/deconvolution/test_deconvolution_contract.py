from inv_framework.deconvolution.api import DeconvolutionTaskRequest, run_deconvolution_task, list_capabilities


def test_discovery_and_incomplete_request():
    caps = list_capabilities()
    assert len(caps["algorithms"]) == 7
    result = run_deconvolution_task(DeconvolutionTaskRequest("blind_dcp", {}, {}))
    assert result.status == "failed"
    assert result.error["code"] == "invalid_parameters"
    assert result.outputs == {}
    assert result.effective_params == {}
