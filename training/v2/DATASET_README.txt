Hover Logix detection dataset v2 (YOLO format), built 2026-11-09
Classes: 0 pallet, 1 pallet_label, 2 location_label, 3 beam, 4 upright, 5 empty_location, 6 pallet_base
Sources (prefix):
  dims_  DroneIMS warehouse photos (984). test = 36 hand-labelled; val = same rack locations + 8%; rest auto-labelled
         by model v1 (full image + 2 overlapping tiles), empty_location from the rack-gap heuristic.
  pd_    Roboflow "pallet detection" v5+v6 (CC BY 4.0, sundharesan-kumaresan): pallet bases -> pallet_base.
         Split by source photo (v5/v6 copies of one photo always in the same split).
  wod_   Roboflow "Warehouse Object Detection" (CC BY 4.0, iptechview): Product-Pallet -> pallet.
Dropped: Rack Detection V2 (lab tube racks), pallet detection v1 (crops), Warehouse Pallet Dataset (unreliable boxes).
