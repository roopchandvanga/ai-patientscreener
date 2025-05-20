from agent import agent

while True:
    query = input("Ask something (or type 'exit'): ")
    if query.lower() == "exit":
        break
    response = agent.run(query)
    print("\n" + response)
