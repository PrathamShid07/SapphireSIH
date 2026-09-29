import math

class CoordinateSystem:
    def __init__(self, ref_lat=None, ref_lon=None, ref_alt=None):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt if ref_alt is not None else 0.0
        
        if ref_lat is not None and ref_lon is not None:
            self.georeferenced = True
            self.crs_name = "LOCAL_ENU_WGS84"
        else:
            self.georeferenced = False
            self.crs_name = "UNGEOLOCATED_LOCAL"

    def latlon_to_local_enu(self, lat, lon, alt=0.0):
        if not self.georeferenced or lat is None or lon is None:
            return None, None, None
            
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(self.ref_lat)
        ref_lon_rad = math.radians(self.ref_lon)
        
        r_earth = 6378137.0
        
        dlat = lat_rad - ref_lat_rad
        dlon = lon_rad - ref_lon_rad
        
        east = dlon * math.cos(ref_lat_rad) * r_earth
        north = dlat * r_earth
        up = alt - self.ref_alt
        
        return round(east, 3), round(north, 3), round(up, 3)

    def to_dict(self):
        return {
            "georeferenced": self.georeferenced,
            "crs_name": self.crs_name,
            "origin": {
                "latitude": self.ref_lat,
                "longitude": self.ref_lon,
                "altitude": self.ref_alt,
            }
        }
