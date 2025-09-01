


# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'How many accidents occurred in 2024?' QUERY: MATCH (a:Accident) WHERE a.Date >= '2024-01-01' AND a.Date <= '2024-12-31' RETURN COUNT(a)",
    "USER INPUT: 'How many drivers involved in more than one accident?' QUERY: MATCH (d:Driver)-[:INVOLVED_AT]->(a:Accident) WITH d, COUNT(a) AS accidentsInvolved WHERE accidentsInvolved > 1 RETURN COUNT(d)",
    "USER INPUT: 'Analyze the drivers involved in the most accidents' QUERY: MATCH (d:Driver)-[:INVOLVED_AT]->(a:Accident) WITH d, COUNT(a) AS accidentsInvolved ORDER BY accidentsInvolved DESC RETURN d.FirstName + ' ' + d.LastName, accidentsInvolved",
    "USER INPUT: 'Analyze the car models involved in the most accidents' QUERY: MATCH (c:Car)-[:INVOLVED_AT]->(a:Accident) WITH c.MakeAndModel AS makeAndModel, COUNT(a) AS totalAccidents RETURN makeAndModel, totalAccidents ORDER BY totalAccidents DESC",
    "USER INPUT: 'Analyze What car model was involved in the most accidents?' QUERY: MATCH (c:Car)-[:INVOLVED_AT]->(a:Accident) WITH c.MakeAndModel AS makeAndModel, COUNT(a) AS totalAccidents RETURN makeAndModel, totalAccidents ORDER BY totalAccidents DESC LIMIT 1"
]