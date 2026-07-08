import sys
from agent import answer_question

def main():
    queries = [
        # q1
        'My email is "rob.brown@email.net" and my password is "OutdoorLife456". I already bought $1700 in products this year. If I buy "3M Littmann Cardiology IV Stethoscope, Standard-Finish Chestpiece, Rose Pink Tube, 27 inch, 6159" will that make me a premier customer?',
        # q2
        'What is the difference between "Digital Thermometer Kit - Item Number 01-415EA" and "HEALTHMAX Digital Forehead Thermometer, Medical Infrared Baby Thermometer for Fever Kids/Adult with Ear Function Body Basal Thermometers Accurate Reading Medically Proven, FDA and CE Approved" and "5 X Fermometer Adhesive Strip Thermometer"?',
        # q3
        'What scales can I buy that cost less than $100 and gives readings in both pounds and kilograms?',
        # q4
        'What items do you carry that can help me loose weight that are not related to exercise?',
        # q5
        'My email is "david.taylor@student.edu" and my password is "Adventure789#". Are there any products on sale that match my interests?',
        # q6
        'My friend is Michelle Garcia, who is one of your customers. Based upon her interests, what do you think I should buy for her?',
        # q7
        'Do you carry any doctor recommended soaps that do not dry out the skin? If you have one, how long (how many days) should one container last, assuming I wash my hands about 4 times a day?',
        # q8
        'Do you have any digital thermometers that comparable to the brand LHCER but are cheaper?',
        # q9
        'I am going snowboarding. What top 2 items would you recommend that are currently available that can help me be safe when snowboarding?',
        # q10
        'I can pay at most $11.50 for a set wrist weights. Do you have any I can buy? My email is "amanda.wilson@gmail.com" and my password is "HealthTracker789".'
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n====================\nQ{i}: {q}\n--------------------")
        try:
            ans = answer_question(q)
            print(f"A{i}: {ans}")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == '__main__':
    main()
