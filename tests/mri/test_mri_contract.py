from inv_framework.mri.api import MRITaskRequest, run_mri_task, list_capabilities


def test_discovery_and_incomplete_request():
    caps = list_capabilities()
    assert len(caps["algorithms"]) == 7
    result = run_mri_task(MRITaskRequest("varnet", {}, {}))
    assert result.status == "failed"
    assert result.error["code"] == "invalid_parameters"
    assert result.outputs == {}
    assert result.effective_params == {}
