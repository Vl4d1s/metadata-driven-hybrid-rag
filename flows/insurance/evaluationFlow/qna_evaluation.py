from evaluations.context_recall import evaluate_context_recall
from tools.rag.graph.tool import graph_rag_tool



def qna_evaluation(question: str, ground_truth: str):
    answer, contexts = graph_rag_tool(question, return_context=True)
    evaluate_context_recall(question, answer, ground_truth, contexts)





emergancy_question = "Will I get a replacement car if mine is stolen and not recovered?"
emergancy_ground_truth = "Yes, you will get a replacement car for up to 10 days if yours is stolen and not recovered."
Accidents_Claims_question = "What should I do immediately after a car accident?"
Accidents_Claims_ground_truth = """
After a car accident, you should:

Note the registration of vehicles involved and exchange details.

Do not admit liability.

Report the accident to the Gardaí immediately (within 24 hours if injuries occur).

Record injuries, damages, and make a diagram/photos of the scene.

Obtain names and addresses of witnesses.

Notify Zurich within 48 hours via the 24 Hour Emergency Helpline
"""
Coverage_Benefits_question = "Does the policy cover medical expenses after an accident?"
Coverage_Benefits_ground_truth = """
Yes, the policy covers medical expenses. The insurer will reimburse the cost of medical treatment for the insured or any occupant of the insured vehicle for bodily injury caused by violent accidental external and visible means in direct connection with the insured vehicle. Coverage is limited to €200 per person injured and only applies to Comprehensive cover.
 """

Coverage_Benefits_question2 = "Does the policy cover child care accessories such as car seats or buggies?"
Coverage_Benefits_ground_truth2 = """
Yes, the policy covers child care accessories. The insurer will pay up to €550 for loss of or damage to any child's push chair, buggy, carrycot or car seat caused by accident, fire, theft or attempted theft following forcible entry. When the vehicle is unattended, accessories (excluding fitted car seat) must be concealed in a locked boot. This applies only to Comprehensive cover.
"""
Conditions_question = "Am I covered if I drive under the influence of alcohol or drugs?"
Conditions_ground_truth = """
No, you are not covered if you drive under the influence of alcohol or drugs. The insurer shall not be liable for any loss or damage if the insured or any insured driver are convicted of driving under the influence of alcohol or drugs. If convicted following a road traffic accident, the insurer is entitled to recover all monies paid in respect of any loss or claim arising from that accident.
"""
Discounts_question = "What happens to my no-claims discount if I make one fire or theft claim?"
Discounts_ground_truth = """
Your no-claims discount will be preserved if you make one fire or theft claim. The discount presently applying will not be stepped back at the next renewal, though the percentage will not be increased at your next renewal date.
"""
Data_Protection_question = "What personal data does the company collect under this policy?"
Data_Protection_ground_truth = """
The company collects contact and identifying information, financial information, employment details, medical and health details including personal habits, sensitive information including criminal convictions and penalty points, information about the insured risk, and claims data including incident circumstances and relevant financial, medical and health information.
"""
Complaints_question = "How can I make a complaint about the company service?"
Complaints_ground_truth = """
First contact your broker if you arranged the policy through one. If unresolved, contact Zurich at (01) 6670666 or write to Customer Service Co-ordinator at Zurich Insurance, PO Box 78, Wexford, or email customercare@zurich.ie. If still unresolved, contact the Financial Services and Pensions Ombudsman, Central Bank of Ireland, or Insurance Ireland.
"""
