import pybgpstream
stream = pybgpstream.BGPStream(
    from_time="2025-10-14 00:00:00", until_time="2025-10-14 00:10:00 UTC",
    collectors=["route-views.sg", "route-views.eqix"],
    record_type="updates"
)

for elem in stream:
    print(elem)

stream.set_data_interface_option()

# update|A|1760400399.112468|routeviews|route-views.sg|None|None|7713|2001:de8:4::7713:1|2a11:5ec0::/48|2001:de8:4::7713:1|7713 16509||None|None