import sys
sys.path.insert(0, 'C:\\DEV\\AI_Projects\\metadata-driven-hybrid-rag')
from evaluations.context_recall import evaluate_context_recall
from evaluations.context_precision import evaluate_context_precision
from tools.rag.graph.tool import graph_rag_tool
from evaluations.faithfulness import evaluate_faithfulness  



def qna_evaluation(question: str, ground_truth: str):
    answer, contexts = graph_rag_tool(question, return_context=True)
    print("ANSWER:", answer)
    contexts = [str(context) for context in contexts]
    evaluate_context_recall(question, answer, ground_truth, contexts)
    evaluate_context_precision(question, answer, ground_truth, contexts)
    evaluate_faithfulness(question, answer, contexts)





# emergancy_question = "Will I get a replacement car if mine is stolen and not recovered?"
# emergancy_ground_truth = "Yes, you will get a replacement car for up to 10 days if yours is stolen and not recovered."
# qna_evaluation(emergancy_question, emergancy_ground_truth)

# Accidents_Claims_question = "What should I do immediately after a car accident?"
# Accidents_Claims_ground_truth = """
# After a car accident, you should:

# Note the registration of vehicles involved and exchange details.

# Do not admit liability.

# Report the accident to the Gardaí immediately (within 24 hours if injuries occur).

# Record injuries, damages, and make a diagram/photos of the scene.

# Obtain names and addresses of witnesses.

# Notify Zurich within 48 hours via the 24 Hour Emergency Helpline
# """
# qna_evaluation(Accidents_Claims_question, Accidents_Claims_ground_truth)
# Coverage_Benefits_question = "Does the policy cover medical expenses after an accident?"
# Coverage_Benefits_ground_truth = """
# Yes, the policy covers medical expenses. The insurer will reimburse the cost of medical treatment for the insured or any occupant of the insured vehicle for bodily injury caused by violent accidental external and visible means in direct connection with the insured vehicle. Coverage is limited to €200 per person injured and only applies to Comprehensive cover.
#  """
# qna_evaluation(Coverage_Benefits_question, Coverage_Benefits_ground_truth)

# Coverage_Benefits_question2 = "Does the policy cover child care accessories such as car seats or buggies?"
# Coverage_Benefits_ground_truth2 = """
# Yes, the policy covers child care accessories. The insurer will pay up to €550 for loss of or damage to any child's push chair, buggy, carrycot or car seat caused by accident, fire, theft or attempted theft following forcible entry. When the vehicle is unattended, accessories (excluding fitted car seat) must be concealed in a locked boot. This applies only to Comprehensive cover.
# """
# qna_evaluation(Coverage_Benefits_question2, Coverage_Benefits_ground_truth2)
# Conditions_question = "Am I covered if I drive under the influence of alcohol or drugs?"
# Conditions_ground_truth = """
# No, you are not covered if you drive under the influence of alcohol or drugs. The insurer shall not be liable for any loss or damage if the insured or any insured driver are convicted of driving under the influence of alcohol or drugs. If convicted following a road traffic accident, the insurer is entitled to recover all monies paid in respect of any loss or claim arising from that accident.
# """
# qna_evaluation(Conditions_question, Conditions_ground_truth)
# Data_Protection_question = "What personal data does the company collect under this policy?"
# Data_Protection_ground_truth = """
# The company collects contact and identifying information, financial information, employment details, medical and health details including personal habits, sensitive information including criminal convictions and penalty points, information about the insured risk, and claims data including incident circumstances and relevant financial, medical and health information.
# """
# qna_evaluation(Data_Protection_question, Data_Protection_ground_truth)
# Complaints_question = "How can I make a complaint about the company service?"
# Complaints_ground_truth = """
# First contact your broker if you arranged the policy through one. If unresolved, contact Zurich at (01) 6670666 or write to Customer Service Co-ordinator at Zurich Insurance, PO Box 78, Wexford, or email customercare@zurich.ie. If still unresolved, contact the Financial Services and Pensions Ombudsman, Central Bank of Ireland, or Insurance Ireland.
# """
# qna_evaluation(Complaints_question, Complaints_ground_truth)


# Emergency Assistance section (page 4) and General Conditions section (page 21)
# two_far_chunks_question = "If I have an accident and need to make a claim, what is the emergency helpline number and how long do I have to notify the insurer?"
# two_far_chunks_ground_truth = "The emergency helpline number is 1890 208 408. You have 48 hours to notify the insurer of the accident."
# qna_evaluation(two_far_chunks_question, two_far_chunks_ground_truth)


# Emergency Assistance section and Section 3, Additional Benefit 2
# two_far_chunks_question2 = "What company issued this policy, and what happens if I don't pay my premium instalments on time?"
# two_far_chunks_ground_truth2 = "The policy was issued by Zurich Insurance plc. If you do not pay your premium instalments on time, any default in payment on the due date will automatically terminate the Policy cover immediately from the date of such default.The policy was issued by Zurich Insurance plc. If you do not pay your premium instalments on time, any default in payment on the due date will automatically terminate the Policy cover immediately from the date of such default."
# qna_evaluation(two_far_chunks_question2, two_far_chunks_ground_truth2)





#  Entity

# driver_basics_question = "what accidents Sarah Mitchell involved in?"
# driver_basics_ground_truth = """
# Sarah Mitchell was involved in the following incident:

# Accident ID: ACC-004

# Description: Sarah Mitchell was traveling north on Main Street and had stopped at a red light when James Rodriguez, distracted by his phone GPS, collided with her vehicle after failing to stop in time.

# Date: November 15, 2024

# Location: Intersection of Main Street and 5th Avenue, Chicago

# Details: Mitchell reported neck pain and was assessed by EMS. The incident was documented in a police report filed by the Chicago Police Department.
# """
# qna_evaluation(driver_basics_question, driver_basics_ground_truth)


# multiple_accidents_question = "What accidents where police reports filed by the Chicago Police Department?"
# multiple_accidents_ground_truth = """
#  Accident ID: ACC-004
#     ◦ Description: Sarah Mitchell was stopped at a red light. James Rodriguez, while using his phone GPS, approached the intersection and struck Mitchell's vehicle from behind at approximately 20 mph.
#     ◦ Date: November 15, 2024
#     ◦ Location: Intersection of Main Street and 5th Avenue, Chicago
#     ◦ Police Report: Yes, Chicago Police Department, Report #2024-11-15-4782
#  Accident ID: ACC-006
#     ◦ Description: Michael Harrison was accelerating through a yellow light, and Ashley Johnson began a left turn, believing Harrison would stop. Johnson's vehicle struck Harrison's passenger side front door in the center of the intersection.
#     ◦ Date: December 3, 2024
#     ◦ Location: Intersection of Roosevelt Road and Halsted Street, Chicago
#     ◦ Police Report: Yes, Chicago Police Department, Report #2024-12-03-8745
#  Accident ID: ACC-007
#     ◦ Description: Oscar Martinez made an illegal U-turn in the middle of the block, crossing into Patricia Nguyen's travel lane. Nguyen's front bumper struck Martinez's passenger door. Martinez received a citation for the illegal U-turn.
#     ◦ Date: February 7, 2025
#     ◦ Location: Commercial Street between 10th and 11th Avenue, Chicago
#     ◦ Police Report: Yes, Chicago Police Department, Report #2025-02-07-667
# """
# qna_evaluation(multiple_accidents_question, multiple_accidents_ground_truth)



# how_many_accidents_question = "how many accidents Ashley Johnson involved at? detail the accidents"
# how_many_accidents_ground_truth = """
# 1. Accident ID: ACC-001
#     ◦ Date: February 8, 2025
#     ◦ Location: Intersection of Westheimer Road and Montrose Boulevard, Houston
#     ◦ Description: Ashley Johnson was driving eastbound and attempted to proceed through a yellow light as she entered the intersection. Her Nissan Altima struck Brian O’Connor’s Chevrolet Malibu, which was turning left. The impact pushed O’Connor’s vehicle into Daniel Rivera’s Ford Explorer. Johnson's vehicle sustained heavy front-end damage, and she complained of minor shoulder pain.
#     ◦ Police Report: Yes, by Houston Police Department, Report #2025-02-08-5678.
# 2. Accident ID: ACC-006
#     ◦ Date: December 3, 2024
#     ◦ Location: Intersection of Roosevelt Road and Halsted Street, Chicago
#     ◦ Description: Ashley Johnson was in the westbound left turn lane and began her left turn on a yellow light, believing Michael Harrison would stop. Harrison was accelerating through the yellow light, and Johnson's vehicle collided with Harrison's passenger side front door in the center of the intersection at approximately 25 mph.
#     ◦ Injuries: Johnson was transported to Rush Medical Center with left arm pain and a possible concussion.
#     ◦ Police Report: Yes, by Chicago Police Department, Report #2024-12-03-8745
# """
# qna_evaluation(how_many_accidents_question, how_many_accidents_ground_truth)

