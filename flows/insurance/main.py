
from data.models.accident_case import AccidentCase
from pydantic import BaseModel
import uuid


from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')
from tools._functions.llama_extractor import extract_file

# FILES_PATH = "C:\DEV\AI_Projects\metadata-driven-hybrid-rag\workflows\insurance\data\reports"
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever , VectorCypherRetriever , Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG




# def load_file_to_db(file_path: str , schema: BaseModel ,extract_name: str="file-parser"):
#     file_data : AccidentCase = extract_file(file_path,schema,extract_name)
    
#     short_uuid = str(uuid.uuid4())[:8]
#     accident_id = f"ACC-{short_uuid}"
#     driver_information = file_data['driver_information']
#     owner_information = file_data['owner_information'] 
#     other_driver_1_information = file_data['other_driver_1_information']
#     other_driver_2_information = file_data['other_driver_2_information']
#     other_driver_3_information = file_data['other_driver_3_information']
#     accident_information = file_data['accident_information']

#     query = f''' '''
#     driver = {
#         'first_name': driver_information.get('first_name'),
#         'last_name': driver_information.get('last_name'),
#         'id_number': driver_information.get('id_number'),
#         'phone': driver_information.get('phone'),
#         'date_of_birth': driver_information.get('date_of_birth'),
#         'zip_code': driver_information.get('zip_code'),
#         'city': driver_information.get('city'),
#         'street': driver_information.get('street'),
#         'house_number': driver_information.get('house_number'),
#         'license_number': driver_information.get('license_number'),
#     }
#     driver_query = f'''MERGE (d:Driver {{IdNumber: "{driver['id_number']}"}})
#         SET d.FirstName = "{driver['first_name']}"
#         SET d.LastName = "{driver['last_name']}"
#         SET d.Phone = "{driver['phone']}"
#         SET d.DateOfBirth = "{driver['date_of_birth']}"
#         SET d.ZipCode = "{driver['zip_code']}"
#         SET d.City = "{driver['city']}"
#         SET d.Street = "{driver['street']}"
#         SET d.HouseNumber = "{driver['house_number']}"
#         SET d.LicenseNumber = "{driver['license_number']}"
#         SET d.embedding = $driver_embedding
#     '''
#     driver_text_for_embedding = (
#         f"Driver named {driver.get('first_name')} {driver.get('last_name')} "
#         f"with ID {driver.get('id_number')} and license {driver.get('license_number')}. "
#         f"Born on {driver.get('date_of_birth')} and lives at "
#         f"{driver.get('house_number')} {driver.get('street')}, "
#         f"{driver.get('city')}, {driver.get('zip_code')}. "
#         f"Drives a {driver.get('vehicle_year', 'unknown year')} {driver.get('vehicle_make_and_model', 'unknown make and model')}."
#     )

#     driver_embedding = None
#     try:
#         embedder = OpenAIEmbeddings(model="text-embedding-ada-002") # Initializes an embedder [4]
#         driver_embedding = embedder.embed_query(driver_text_for_embedding) # Generates the embedding vector [7]
#         print(f"Generated embedding of dimension for main driver: {len(driver_embedding)}")
#     except Exception as e:
#         print(f"Error generating embedding for main driver: {e}. Embedding will not be added for this driver.")

#     # Collect query parameters for embeddings
#     query_params = {"driver_embedding": driver_embedding}


#     car1 = {
#         'license_plate': driver_information.get('license_plate'),
#         'vehicle_make_and_model': driver_information.get('vehicle_make_and_model'),
#         'vehicle_year': driver_information.get('vehicle_year'),
#     }

#     car1_query = f'''MERGE (car1:Car {{LicensePlate: "{car1['license_plate']}"}})
# SET car1.MakeAndModel = "{car1['vehicle_make_and_model']}"
# SET car1.Year = "{car1['vehicle_year']}"
# SET car1.embedding = $car1_embedding'''

#     # Create car1 embedding
#     car1_text_for_embedding = (
#         f"Car with license plate {car1.get('license_plate')} "
#         f"is a {car1.get('vehicle_year', 'unknown year')} "
#         f"{car1.get('vehicle_make_and_model', 'unknown make and model')}"
#     )
#     car1_embedding = None
#     try:
#         embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#         car1_embedding = embedder.embed_query(car1_text_for_embedding)
#         print(f"Generated embedding of dimension for car1: {len(car1_embedding)}")
#     except Exception as e:
#         print(f"Error generating embedding for car1: {e}. Embedding will not be added for this car.")
#     query_params["car1_embedding"] = car1_embedding
#     owner = {
#         'first_name': owner_information.get('first_name'),
#         'last_name': owner_information.get('last_name'),
#         'id_number': owner_information.get('id_number'),
#         'phone': owner_information.get('phone'),
#         'date_of_birth': owner_information.get('date_of_birth'),
#         'zip_code': owner_information.get('zip_code'),
#         'city': owner_information.get('city'),
#         'street': owner_information.get('street'),
#         'house_number': owner_information.get('house_number','NULL'),
#     }

#     owner_query = f'''MERGE (o:Owner {{IdNumber: "{owner['id_number']}"}})
# SET o.FirstName = "{owner['first_name']}"
# SET o.LastName = "{owner['last_name']}"
# SET o.Phone = "{owner['phone']}"
# SET o.DateOfBirth = "{owner['date_of_birth']}"
# SET o.ZipCode = "{owner['zip_code']}"
# SET o.City = "{owner['city']}"
# SET o.Street = "{owner['street']}"
# SET o.HouseNumber = "{owner['house_number']}"
# SET o.embedding = $owner_embedding
#     '''

#     # Create owner embedding
#     owner_text_for_embedding = (
#         f"Owner named {owner.get('first_name')} {owner.get('last_name')} "
#         f"with ID {owner.get('id_number')} and phone {owner.get('phone')}. "
#         f"Born on {owner.get('date_of_birth')} and lives at "
#         f"{owner.get('house_number')} {owner.get('street')}, "
#         f"{owner.get('city')}, {owner.get('zip_code')}."
#     )
#     owner_embedding = None
#     try:
#         embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#         owner_embedding = embedder.embed_query(owner_text_for_embedding)
#         print(f"Generated embedding of dimension for owner: {len(owner_embedding)}")
#     except Exception as e:
#         print(f"Error generating embedding for owner: {e}. Embedding will not be added for this owner.")
#     query_params["owner_embedding"] = owner_embedding

#     accident = {
#         'date': accident_information.get('date'),
#         'time': accident_information.get('time'),
#         'city': accident_information.get('city'),
#         'location': accident_information.get('location'),
#         'number_of_vehicles_involved': accident_information.get('number_of_vehicles_involved'),
#         'police_report_made': accident_information.get('police_report_made'),
#         'police_agency': accident_information.get('police_agency'),
#         'description': accident_information.get('description'),
#     }

#     accident_query = f'''MERGE (a:Accident {{Id: "{accident_id}"}})
# SET a.Date = "{accident['date']}"
# SET a.Time = "{accident['time']}"
# SET a.City = "{accident['city']}"
# SET a.Location = "{accident['location']}"
# SET a.NumberOfVehiclesInvolved = "{accident['number_of_vehicles_involved']}"
# SET a.PoliceReportMade = "{accident['police_report_made']}"
# SET a.PoliceAgency = "{accident['police_agency']}"
# SET a.Description = "{accident['description']}"
# SET a.embedding = $accident_embedding'''

#     # Create accident embedding
#     accident_text_for_embedding = (
#         f"Accident on {accident.get('date')} at {accident.get('time')} in {accident.get('city')} "
#         f"at {accident.get('location')}. "
#         f"Vehicles involved: {accident.get('number_of_vehicles_involved')}. "
#         f"Police report: {accident.get('police_report_made')} by {accident.get('police_agency')}. "
#         f"Description: {accident.get('description')}"
#     )
#     accident_embedding = None
#     try:
#         embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#         accident_embedding = embedder.embed_query(accident_text_for_embedding)
#         print(f"Generated embedding of dimension for accident: {len(accident_embedding)}")
#     except Exception as e:
#         print(f"Error generating embedding for accident: {e}. Embedding will not be added for this accident.")
#     query_params["accident_embedding"] = accident_embedding
    

#     if other_driver_1_information and other_driver_1_information.get('id_number') != '':
#         other_driver_1 = {
#         'first_name': other_driver_1_information.get('first_name'),
#         'last_name': other_driver_1_information.get('last_name'),
#         'id_number': other_driver_1_information.get('id_number'),
#         'phone': other_driver_1_information.get('phone'),
#         'date_of_birth': other_driver_1_information.get('date_of_birth'),
#         'zip_code': other_driver_1_information.get('zip_code'),
#         'city': other_driver_1_information.get('city'),
#         'street': other_driver_1_information.get('street'),
#         'house_number': other_driver_1_information.get('house_number'),
#         'license_number': other_driver_1_information.get('license_number'),
#         }
#         other_driver_1_query = f'''MERGE (od1:Driver {{IdNumber: "{other_driver_1['id_number']}"}})
# SET od1.FirstName = "{other_driver_1['first_name']}"
# SET od1.LastName = "{other_driver_1['last_name']}"
# SET od1.Phone = "{other_driver_1['phone']}"
# SET od1.DateOfBirth = "{other_driver_1['date_of_birth']}"
# SET od1.ZipCode = "{other_driver_1['zip_code']}"
# SET od1.City = "{other_driver_1['city']}"
# SET od1.Street = "{other_driver_1['street']}"
# SET od1.HouseNumber = "{other_driver_1['house_number']}"
# SET od1.LicenseNumber = "{other_driver_1['license_number']}"
# SET od1.embedding = $other_driver_1_embedding'''

#         # Create other_driver_1 embedding
#         other_driver_1_text_for_embedding = (
#             f"Driver named {other_driver_1.get('first_name')} {other_driver_1.get('last_name')} "
#             f"with ID {other_driver_1.get('id_number')} and license {other_driver_1.get('license_number')}. "
#             f"Born on {other_driver_1.get('date_of_birth')} and lives at "
#             f"{other_driver_1.get('house_number')} {other_driver_1.get('street')}, "
#             f"{other_driver_1.get('city')}, {other_driver_1.get('zip_code')}"
#         )
#         other_driver_1_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             other_driver_1_embedding = embedder.embed_query(other_driver_1_text_for_embedding)
#             print(f"Generated embedding of dimension for other_driver_1: {len(other_driver_1_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for other_driver_1: {e}. Embedding will not be added for this driver.")
#         query_params["other_driver_1_embedding"] = other_driver_1_embedding

#         car2 = {
#             'license_plate': other_driver_1_information.get('license_plate'),
#             'vehicle_make_and_model': other_driver_1_information.get('vehicle_make_and_model'),
#             'vehicle_year': other_driver_1_information.get('vehicle_year'),
#         }

#         car2_query = f'''MERGE (car2:Car {{LicensePlate: "{car2['license_plate']}"}})
# SET car2.MakeAndModel = "{car2['vehicle_make_and_model']}"
# SET car2.Year = "{car2['vehicle_year']}"
# SET car2.embedding = $car2_embedding'''

#         # Create car2 embedding
#         car2_text_for_embedding = (
#             f"Car with license plate {car2.get('license_plate')} "
#             f"is a {car2.get('vehicle_year', 'unknown year')} "
#             f"{car2.get('vehicle_make_and_model', 'unknown make and model')}"
#         )
#         car2_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             car2_embedding = embedder.embed_query(car2_text_for_embedding)
#             print(f"Generated embedding of dimension for car2: {len(car2_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for car2: {e}. Embedding will not be added for this car.")
#         query_params["car2_embedding"] = car2_embedding

#     if other_driver_2_information:
#         other_driver_2 = {
#         'first_name': other_driver_2_information.get('first_name'),
#         'last_name': other_driver_2_information.get('last_name'),
#         'id_number': other_driver_2_information.get('id_number'),
#         'phone': other_driver_2_information.get('phone'),
#         'date_of_birth': other_driver_2_information.get('date_of_birth'),
#         'zip_code': other_driver_2_information.get('zip_code'),
#         'city': other_driver_2_information.get('city'),
#         'street': other_driver_2_information.get('street'),
#         'house_number': other_driver_2_information.get('house_number'),
#         'license_number': other_driver_2_information.get('license_number'),
#         }
#         other_driver_2_query = f'''MERGE (od2:Driver {{IdNumber: "{other_driver_2['id_number']}"}})
# SET od2.FirstName = "{other_driver_2['first_name']}"
# SET od2.LastName = "{other_driver_2['last_name']}"
# SET od2.Phone = "{other_driver_2['phone']}"
# SET od2.DateOfBirth = "{other_driver_2['date_of_birth']}"
# SET od2.ZipCode = "{other_driver_2['zip_code']}"
# SET od2.City = "{other_driver_2['city']}"
# SET od2.Street = "{other_driver_2['street']}"
# SET od2.HouseNumber = "{other_driver_2['house_number']}"
# SET od2.LicenseNumber = "{other_driver_2['license_number']}"
# SET od2.embedding = $other_driver_2_embedding'''

#         # Create other_driver_2 embedding
#         other_driver_2_text_for_embedding = (
#             f"Driver named {other_driver_2.get('first_name')} {other_driver_2.get('last_name')} "
#             f"with ID {other_driver_2.get('id_number')} and license {other_driver_2.get('license_number')}. "
#             f"Born on {other_driver_2.get('date_of_birth')} and lives at "
#             f"{other_driver_2.get('house_number')} {other_driver_2.get('street')}, "
#             f"{other_driver_2.get('city')}, {other_driver_2.get('zip_code')}"
#         )
#         other_driver_2_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             other_driver_2_embedding = embedder.embed_query(other_driver_2_text_for_embedding)
#             print(f"Generated embedding of dimension for other_driver_2: {len(other_driver_2_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for other_driver_2: {e}. Embedding will not be added for this driver.")
#         query_params["other_driver_2_embedding"] = other_driver_2_embedding

#         car3 = {
#             'license_plate': other_driver_2_information.get('license_plate'),
#             'vehicle_make_and_model': other_driver_2_information.get('vehicle_make_and_model'),
#             'vehicle_year': other_driver_2_information.get('vehicle_year'),
#         }

#         car3_query = f'''MERGE (car3:Car {{LicensePlate: "{car3['license_plate']}"}})
# SET car3.MakeAndModel = "{car3['vehicle_make_and_model']}"
# SET car3.Year = "{car3['vehicle_year']}"
# SET car3.embedding = $car3_embedding'''

#         # Create car3 embedding
#         car3_text_for_embedding = (
#             f"Car with license plate {car3.get('license_plate')} "
#             f"is a {car3.get('vehicle_year', 'unknown year')} "
#             f"{car3.get('vehicle_make_and_model', 'unknown make and model')}"
#         )
#         car3_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             car3_embedding = embedder.embed_query(car3_text_for_embedding)
#             print(f"Generated embedding of dimension for car3: {len(car3_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for car3: {e}. Embedding will not be added for this car.")
#         query_params["car3_embedding"] = car3_embedding

#     if other_driver_3_information:
#         other_driver_3 = {
#         'first_name': other_driver_3_information.get('first_name'),
#         'last_name': other_driver_3_information.get('last_name'),
#         'id_number': other_driver_3_information.get('id_number'),
#         'phone': other_driver_3_information.get('phone'),
#         'date_of_birth': other_driver_3_information.get('date_of_birth'),
#         'zip_code': other_driver_3_information.get('zip_code'),
#         'city': other_driver_3_information.get('city'),
#         'street': other_driver_3_information.get('street'),
#         'house_number': other_driver_3_information.get('house_number'),
#         'license_number': other_driver_3_information.get('license_number'),
#         }
#         other_driver_3_query = f'''MERGE (od3:Driver {{IdNumber: "{other_driver_3['id_number']}"}})
# SET od3.FirstName = "{other_driver_3['first_name']}"
# SET od3.LastName = "{other_driver_3['last_name']}"
# SET od3.Phone = "{other_driver_3['phone']}"
# SET od3.DateOfBirth = "{other_driver_3['date_of_birth']}"
# SET od3.ZipCode = "{other_driver_3['zip_code']}"
# SET od3.City = "{other_driver_3['city']}"
# SET od3.Street = "{other_driver_3['street']}"
# SET od3.HouseNumber = "{other_driver_3['house_number']}"
# SET od3.LicenseNumber = "{other_driver_3['license_number']}"
# SET od3.embedding = $other_driver_3_embedding'''

#         # Create other_driver_3 embedding
#         other_driver_3_text_for_embedding = (
#             f"Driver named {other_driver_3.get('first_name')} {other_driver_3.get('last_name')} "
#             f"with ID {other_driver_3.get('id_number')} and license {other_driver_3.get('license_number')}. "
#             f"Born on {other_driver_3.get('date_of_birth')} and lives at "
#             f"{other_driver_3.get('house_number')} {other_driver_3.get('street')}, "
#             f"{other_driver_3.get('city')}, {other_driver_3.get('zip_code')}"
#         )
#         other_driver_3_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             other_driver_3_embedding = embedder.embed_query(other_driver_3_text_for_embedding)
#             print(f"Generated embedding of dimension for other_driver_3: {len(other_driver_3_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for other_driver_3: {e}. Embedding will not be added for this driver.")
#         query_params["other_driver_3_embedding"] = other_driver_3_embedding

#         car4 = {
#             'license_plate': other_driver_3_information.get('license_plate'),
#             'vehicle_make_and_model': other_driver_3_information.get('vehicle_make_and_model'),
#             'vehicle_year': other_driver_3_information.get('vehicle_year'),
#         }

#         car4_query = f'''MERGE (car4:Car {{LicensePlate: "{car4['license_plate']}"}})
# SET car4.MakeAndModel = "{car4['vehicle_make_and_model']}"
# SET car4.Year = "{car4['vehicle_year']}"
# SET car4.embedding = $car4_embedding'''

#         # Create car4 embedding
#         car4_text_for_embedding = (
#             f"Car with license plate {car4.get('license_plate')} "
#             f"is a {car4.get('vehicle_year', 'unknown year')} "
#             f"{car4.get('vehicle_make_and_model', 'unknown make and model')}"
#         )
#         car4_embedding = None
#         try:
#             embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
#             car4_embedding = embedder.embed_query(car4_text_for_embedding)
#             print(f"Generated embedding of dimension for car4: {len(car4_embedding)}")
#         except Exception as e:
#             print(f"Error generating embedding for car4: {e}. Embedding will not be added for this car.")
#         query_params["car4_embedding"] = car4_embedding
        

#     # create relationships
#     base_relationships = f'''
#         MERGE (d)-[:DRIVED_AT]->(car1)
#         MERGE (o)-[:OWNS]->(car1)
#         MERGE (d)-[:INVOLVED_AT {{title: "PrimaryDriver"}}]->(a)
#         MERGE (car1)-[:INVOLVED_AT {{title: "Car"}}]->(a)
#     '''
#     if other_driver_1_information and other_driver_1_information.get('id_number') != '':
#         base_relationships += f'''
#         MERGE (od1)-[:DRIVED_AT]->(car2)
#         MERGE (od1)-[:INVOLVED_AT {{title: "OtherDriver"}}]->(a)
#         MERGE (car2)-[:INVOLVED_AT {{title: "Car"}}]->(a)
#         '''
#     if other_driver_2_information:
#         base_relationships += f'''
#         MERGE (od2)-[:DRIVED_AT]->(car3)
#         MERGE (od2)-[:INVOLVED_AT {{title: "OtherDriver"}}]->(a)
#         MERGE (car3)-[:INVOLVED_AT {{title: "Car"}}]->(a)
#         '''
#     if other_driver_3_information:
#         base_relationships += f'''
#         MERGE (od3)-[:DRIVED_AT]->(car4)
#         MERGE (od3)-[:INVOLVED_AT {{title: "OtherDriver"}}]->(a)
#         MERGE (car4)-[:INVOLVED_AT {{title: "Car"}}]->(a)
#         '''

    


#     # Build final single query string
#     query_parts = [
#         driver_query,
#         car1_query,
#         owner_query,
#         accident_query,
#     ]

#     if other_driver_1_information and other_driver_1_information.get('id_number') != '':
#         query_parts.append(other_driver_1_query)
#         query_parts.append(car2_query)

#     if other_driver_2_information:
#         query_parts.append(other_driver_2_query)
#         query_parts.append(car3_query)

#     if other_driver_3_information:
#         query_parts.append(other_driver_3_query)
#         query_parts.append(car4_query)

#     query_parts.append(base_relationships)

#     query = "\n".join(q for q in query_parts if q)

#     # Debug: print the final Cypher query
#     print("query:\n", query)

#     # query = """
#     #  MERGE (m:Movie {title: $movie_title})
#     #  MERGE (u:User {name: $user_name})
#     #  MERGE (u)-[r:RATED {rating: $rating}]->(m)
#     #  """
#     print("file_path:",file_path)
#     print("schema:",schema)
#     print("file_data:",file_data)
#     print("driver:", driver)
#     print("car1:", car1)
#     print("owner:", owner)
#     print("accident:", accident)
#     print("other_driver_1:", other_driver_1_information)
#     print("other_driver_2:", other_driver_2_information)
#     print("other_driver_3:", other_driver_3_information)

#     graph_driver = GraphDatabase.driver(
#         os.getenv("NEO4J_URI"), 
#         auth=(
#             os.getenv("NEO4J_USERNAME"), 
#             os.getenv("NEO4J_PASSWORD")
#         )
#     )
#     with graph_driver.session() as session:
#         try:
#             # Execute the query without a separate parameters argument
#             result = session.run(query,parameters=query_params)
#             for record in result:
#                 print(f"Node processed: {record['d']}")
#         except Exception as e:
#             print(f"Error executing query: {e}")
#         finally:
#             graph_driver.close()
#     return

# testing_file_path = "C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag\\flows\\insurance\\data\\reports\\008.pdf"
 
# load_file_to_db(testing_file_path,AccidentCase)