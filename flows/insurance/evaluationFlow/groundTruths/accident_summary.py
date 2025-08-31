question = "Give me a summary about the accident ACC-002"

accident_summary_ground_truth = """
2025-02-03 17:29:00 - SUN_GLARE - Sun directly in drivers' line of sight during sunset
2025-02-03 17:29:45 - SPEED_REDUCTION - Peter Roberts reduced speed from 35 mph to 15 mph due to sun glare
2025-02-03 17:30:00 - ACCIDENT_OCCURRED - Julia Evans struck Peter Roberts' vehicle at approximately 20 mph
2025-02-03 17:30:00 - POLICE_REPORT - Police report made by Chicago Police Department, Report #2025-02-03-445, noting sun glare as a contributing factor
2025-02-03 17:30:00 - VEHICLE_INFO - Both vehicles traveling westbound during sunset; Julia Evans unable to see Roberts' brake lights due to sun glare
2025-02-03 17:30:00 - BRAKE_APPLICATION - Julia Evans noticed stopped traffic and applied brakes but struck Roberts' vehicle
2025-02-03 17:45:00 - INJURY_TRANSPORT - Julia Evans transported to urgent care due to chest pain from airbag deployment
2025-02-03 17:45:00 - AIRBAG_DEPLOYMENT - Julia Evans' airbag deployed upon impact
2025-02-03 17:45:00 - CHEST_PAIN_COMPLAINT - Julia Evans complained of chest pain from airbag deployment
2025-02-03 00:00:00 - DRIVER_INFO - Peter Roberts, DOB: 1977-01-30, License: R345-6789-0123, Phone: (555) 345-6789, Address: 2345 West Chicago Avenue, Chicago, IL 60623
2025-02-03 00:00:00 - VEHICLE_INFO - Peter Roberts' vehicle: 2019 Volkswagen Atlas, License Plate: EFG 2345
2025-02-03 00:00:00 - FOLLOWING_VEHICLE_INFO - Julia Evans was following Peter Roberts at 35 mph, unable to see brake lights due to sun glare
2025-02-03 17:30:00 - NO_INJURIES - Peter Roberts reported no injuries from the accident
2025-02-03 00:00:00 - OTHER_DRIVER_INFO - Julia Evans, DOB: 1983-03-18, License: E456-7890-1234, Phone: (555) 456-7890, Address: 3456 East Oak Street, Chicago, IL 60624
2025-02-03 00:00:00 - OTHER_VEHICLE_INFO - Julia Evans' vehicle: 2020 Infiniti QX60, License Plate: HIJ 3456
"""