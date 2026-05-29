# Rules and guidelines of the Equal Credit Opportunity Act (ECOA - Regulation B)
# This data is used to seed the local RAG Vector Database.

ECOA_REGULATION_B_RULES = [
    {
        "id": "ecoa_1002_4_general",
        "title": "ECOA 12 CFR § 1002.4 - General Rule Prohibiting Discrimination",
        "content": (
            "A creditor shall not discriminate against an applicant on a prohibited basis regarding "
            "any aspect of a credit transaction. Prohibited bases include race, color, religion, "
            "national origin, sex, marital status, age (provided the applicant has the capacity to contract), "
            "because all or part of the applicant's income derives from any public assistance program, "
            "or because the applicant has in good faith exercised any right under the Consumer Credit Protection Act."
        ),
        "category": "General Anti-Discrimination"
    },
    {
        "id": "ecoa_1002_6_age_evaluation",
        "title": "ECOA 12 CFR § 1002.6(b)(2) - Specific Rules Concerning Age Evaluation",
        "content": (
            "Except as otherwise permitted, a creditor shall not take into account an applicant's age "
            "(provided that the applicant has the capacity to enter into a binding contract) or whether an "
            "applicant's income derives from any public assistance program. In an empirically derived, "
            "demonstrably and statistically sound, credit scoring system, a creditor may use age as a "
            "predictive variable, provided that the age of an elderly applicant (62 years of age or older) "
            "is not assigned a negative factor or value, or scored less favorably than other age cohorts."
        ),
        "category": "Age Evaluation Rules"
    },
    {
        "id": "ecoa_1002_6_elderly_definition",
        "title": "ECOA 12 CFR § 1002.2(o) - Definition of Elderly",
        "content": (
            "Elderly means an applicant who is 62 years of age or older. A credit scoring system "
            "must treat applicants who are 62 or older at least as favorably as applicants who are younger "
            "than 62. In a judgmental system of evaluating creditworthiness, a creditor may consider "
            "an applicant's age only for the purpose of determining a pertinent element of creditworthiness, "
            "such as the adequacy of the applicant's income or security at the time of retirement."
        ),
        "category": "Definitions"
    },
    {
        "id": "ecoa_1002_9_adverse_action_notice",
        "title": "ECOA 12 CFR § 1002.9 - Notifications of Adverse Action",
        "content": (
            "A creditor must notify an applicant of action taken within 30 days after receiving a completed application. "
            "An adverse action notice (such as a credit rejection) must be in writing and contain the specific, "
            "principal reasons for the adverse action. Vague or circular statements of reasons (such as 'you did not meet "
            "our standards') do not satisfy the requirements. The reasons must relate directly to credit factors "
            "like credit utilization, payment history, late payments, or income, and must be verifiable."
        ),
        "category": "Adverse Action Notice"
    },
    {
        "id": "ecoa_1002_6_public_assistance",
        "title": "ECOA 12 CFR § 1002.6(b)(5) - Evaluation of Public Assistance Income",
        "content": (
            "A creditor shall not discount or exclude from consideration the income of an applicant because "
            "the income is derived from a public assistance program, a pension, annuity, or insurance benefit. "
            "A creditor may consider the amount and probable continuity of the income in evaluating creditworthiness, "
            "but must treat public assistance income as favorably as other forms of earned income."
        ),
        "category": "Income Evaluation"
    }
]
