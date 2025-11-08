from datetime import datetime
import os
from ripe.atlas.cousteau import (
  Traceroute,
  AtlasSource,
  AtlasCreateRequest,
  AtlasResultsRequest
)

ATLAS_API_KEY = "69812379-3024-4761-960a-07bf45249afb"

traceroute = Traceroute(
    af=4,
    target="www.ripe.net",
    description="testing",
    protocol="ICMP",
)

source = AtlasSource(
    type="area",
    value="WW",
    requested=5,
    tags={"include":["system-ipv4-works"]}
)
source1 = AtlasSource(
    type="country",
    value="NL",
    requested=50,
    tags={"exclude": ["system-anchor"]}
)

atlas_request = AtlasCreateRequest(
    key=ATLAS_API_KEY,
    measurements=[traceroute],
    sources=[source, source1],
    is_oneoff=True
)

# (is_success, response) = atlas_request.create()
# print(response)

kwargs = {
    "msm_id": 136749956
}

is_success, results = AtlasResultsRequest(**kwargs).create()
if is_success:
    print(results)
