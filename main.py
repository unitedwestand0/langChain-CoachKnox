import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI

# from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print("Your XAi API Key", os.environ.get("XAI_API_KEY"))

    boxing_questions = """
    1. What do I need for my first {boxing} session?
    2. Am I too old to start {boxing} ?
    3. What are the main benefits of {boxing} for fitness and health?
    4. Why is {boxing} better than other exercises like running or weightlifting for overall fitness?
    5. What is the first thing a beginner in {boxing} should learn, like stance and footwork?
    6. How do I structure a basic {boxing} training routine as a beginner?
    """

    boxing = """
    For your first boxing session, focus on essentials to protect yourself and train effectively. You'll need boxing gloves 
    (10-12 oz for beginners to provide cushioning), hand wraps (to support wrists and knuckles—aim for 180-inch cotton or Mexican-style), 
    comfortable athletic clothing like shorts or leggings and a moisture-wicking shirt, supportive sneakers with good grip (cross-trainers 
    work if you don't have boxing-specific shoes yet), and a water bottle. Optional but recommended: a mouth guard for safety and a jump rope 
    for warm-ups. Avoid heavy bags or pads initially unless instructed by a trainer.

    No, you're not too old to start boxing at 40 or beyond—many people begin later in life and thrive. Boxing can be adapted for any age with 
    proper guidance, focusing on fitness rather than competition. It improves cardiovascular health, strength, and coordination, but consult a 
    doctor first if you have pre-existing conditions like joint issues or heart problems. Start slow with non-contact classes to build confidence 
    and avoid injury. Masters boxing programs exist specifically for older adults, emphasizing health benefits like better balance and bone density.

    Boxing offers comprehensive benefits: it boosts cardiovascular health by improving heart efficiency and lowering blood pressure; builds full-body 
    strength and muscle definition, especially in the core, arms, and legs; enhances hand-eye coordination, balance, and agility; supports mental health 
    by reducing stress, improving mood through endorphin release, and boosting self-esteem; strengthens bones and joints to combat osteoporosis; and aids 
    in weight management by burning 500-800 calories per hour. It's also a confidence builder and can improve focus and discipline.
    
    Boxing stands out for overall fitness because it combines cardio (like running) with strength training (like weightlifting) in a dynamic, full-body 
    workout that also incorporates coordination, agility, and mental sharpness—elements often missing in isolated exercises. Unlike running's repetitive 
    impact on joints, boxing varies movements to reduce boredom and injury risk while burning more calories through high-intensity intervals. Compared 
    to weightlifting, it adds explosive power, endurance, and real-world functional strength without needing equipment variety. It also provides stress 
    relief and self-defense skills, making it more engaging and holistic for long-term adherence.

    The first thing to master is the basic boxing stance, as it forms the foundation for balance, power, and defense. For orthodox fighters (right-handed), 
    place your left foot forward, feet shoulder-width apart, knees slightly bent, weight on the balls of your feet. Right heel slightly raised, hands up with 
    left guarding your chin and right near your cheek, elbows tucked in. Practice step-drag footwork: step forward with the lead foot and drag the rear, 
    maintaining stance. This ensures stability and quick movement.

    A basic beginner routine could be 3-4 days a week, 45-60 minutes per session: Start with 5-10 minutes of warm-up (jumping rope or shadowboxing); 10-15 
    minutes on technique (practicing stance, jabs, and footwork); 15-20 minutes of bag work or drills (like simple combos); 10 minutes of strength/conditioning 
    (push-ups, squats, planks); and end with 5 minutes of cool-down stretches. Rest days are crucial—include active recovery like walking. Track progress and 
    adjust intensity to avoid burnout.

    """

    summary_boxing_template = PromptTemplate(
        input_variables=["boxing"], template=boxing_questions
    )
    # boxing_llm = ChatOllama(temperature=0, model="gemma3:270m")
    boxing_llm = ChatXAI(temperature=0, model="grok-4-1-fast-reasoning")
    boxing_chain = summary_boxing_template | boxing_llm

    boxing_response = boxing_chain.invoke(input={"boxing": boxing})

    print(boxing_response.content)

    information = """

    Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa. 
    He taught himself programming as a child and sold his first video game at age 12 for ~$500.
    He holds citizenship in South Africa (by birth), Canada (through his mother), and the United States (naturalized in 2002). 
    Musk is known for his first principles thinking: breaking problems down to fundamental truths and reasoning up from there 
    — a mindset that's hugely relevant when building AI agents or chains in LangChain! As of late 2025/early 2026, he is the 
    world's wealthiest person, with a net worth estimated around $717 - 726 billion (mostly from Tesla + SpaceX stakes).
    
    """
    summary_template = """
    Given the information {information} about a person I want you to create :
    1. A short summary 
    2. Two interesting facts about them

    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatXAI(temperature=0, model="grok-4-fast-non-reasoning")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
