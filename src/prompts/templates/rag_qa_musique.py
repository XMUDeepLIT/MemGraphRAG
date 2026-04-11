# from `gold_with_3_distractors_context_cot_qa_codex.txt`

one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)



one_shot_ircot_demo = (
    f'{one_shot_rag_qa_docs}'
    '\n\nQuestion: '
    f"When was Neville A. Stanton's employer founded?"
    '\nThought: '
    f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
    '\n\n'
)


rag_qa_system_with_knowledge = (
    'You are an intelligent reading comprehension assistant. Your task is to analyze the provided text passages and answer the given question through step-by-step reasoning.\n\n'
    'Follow these guidelines:\n'
    '1. First, carefully read and understand all the provided text passages.\n'
    '2. Identify relevant information from the passages that relates to the question.\n'
    '3. If the passages contain sufficient information to answer the question:\n'
    '   - Extract and synthesize the relevant facts\n'
    '   - Build a logical chain of reasoning step by step\n'
    '4. If the passages do NOT contain enough information:\n'
    '   - You may supplement with your own knowledge when appropriate\n'
    '   - Ensure any external knowledge is relevant and accurate\n'
    '5. Structure your response as follows:\n'
    '   - Start your response with "Thought: "\n'
    '   - Present your reasoning process in clear, numbered steps\n'
    '   - Conclude with "Answer: " followed by your final answer\n'
    '6. Be concise but thorough in your reasoning. Show how you connect the information to reach your conclusion.'
)

one_shot_rag_qa_with_knowledge_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "Who directed Finding Nemo and what university did he attend?"
    '\nThought: '
)

one_shot_rag_qa_with_knowledge_output = (
    "Step 1: I need to find information about who directed Finding Nemo from the provided passages.\n"
    "Step 2: From the 'Finding Nemo' passage, I can see it was directed by Andrew Stanton.\n"
    "Step 3: Now I need to find what university Andrew Stanton attended. The provided passages do not contain this information.\n"
    "Step 4: Based on general knowledge, Andrew Stanton attended the University of Southern California (USC) School of Cinematic Arts, where he earned his Bachelor of Fine Arts degree.\n"
    "Answer: Andrew Stanton directed Finding Nemo, and he attended the University of Southern California."
)


rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
