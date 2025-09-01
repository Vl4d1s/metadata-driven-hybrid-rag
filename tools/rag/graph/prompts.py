router_examples = """
Question: "What is the coverage amount for collision damage?"
Classification: policy
Reasoning: Asks about policy terms and coverage details

Question: "What are the deductibles mentioned in the policy?"
Classification: policy
Reasoning: Seeks information about policy terms and conditions

Question: "Tell me about the driver John Smith"
Classification: entity
Reasoning: Asks about a specific person/driver entity

Question: "What cars are involved in the accident?"
Classification: entity
Reasoning: Asks about specific vehicles/car entities

Question: "Show me details about accident case ID 12345"
Classification: entity
Reasoning: Asks about a specific accident entity

Question: "What happened in the accident on Main Street?"
Classification: entity
Reasoning: Asks about a specific accident event/entity

Question: "What is the coverage for the accident that happened on Main Street?"
Classification: both
Reasoning: Asks about policy coverage AND specific accident entity

Question: "Are the vehicles involved in accident 12345 covered under the policy?"
Classification: both
Reasoning: Asks about policy coverage AND specific accident/vehicle entities
"""


retrieval_query_entity = """
RETURN 
  node.Id AS accidentId,
  node.Description AS accidentDescription,
  node.Location AS location,
  node.Date AS date,
  node.Time AS time,
  node.City AS city,
  node.NumberOfVehiclesInvolved AS numberOfVehicles,
  node.PoliceAgency AS policeAgency,
  node.PoliceReportMade AS policeReportMade,
  score AS similarityScore,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(c:Car) 
    OPTIONAL MATCH (c)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH c, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Car',
      makeAndModel: c.MakeAndModel,
      year: c.Year,
      licensePlate: c.LicensePlate,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedCars,
  collect { 
    MATCH (node)<-[:INVOLVED_AT]-(d:Driver) 
    OPTIONAL MATCH (d)-[:INVOLVED_AT]->(otherAcc:Accident)
    WHERE otherAcc <> node
    WITH d, collect(DISTINCT otherAcc) AS otherAccidents
    RETURN {
      nodeType: 'Driver',
      firstName: d.FirstName,
      lastName: d.LastName,
      dateOfBirth: d.DateOfBirth,
      licenseNumber: d.LicenseNumber,
      phone: d.Phone,
      city: d.City,
      street: d.Street,
      houseNumber: d.HouseNumber,
      zipCode: d.ZipCode,
      idNumber: d.IdNumber,
      otherAccidents: [acc IN otherAccidents | {
        accidentId: acc.Id,
        date: acc.Date,
        time: acc.Time,
        location: acc.Location,
        city: acc.City
      }],
      totalAccidents: size(otherAccidents) + 1
    }
  } as involvedDrivers
ORDER BY similarityScore DESC
"""