import socket

def receive_response(s):
    waiting = True
    data = s.recv(1024).decode()
    while waiting: 
        if data.startswith("WAIT"):
            break
        else:
            parts = data.split(',')
            state = [float(x) for x in parts[:-2]]
            reward = int(parts[-2])
            terminal = bool(parts[-1])
            return state, reward, terminal

def send_instruction(s, instruction):
    s.sendall(f"{instruction[0]},{instruction[1]}".encode())
    print(f"Sent instruction: {instruction}")

def main():
    host = '127.0.0.1'
    port = 8888

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("Connected to the server.")
        
        for _ in range(5):  # Repeat the cycle 5 times
            wait_for_action_finished = True
            instruction = (4, 0.5)  # Replace with your actual instruction values
            send_instruction(s, instruction)
            print("sending instruction from python")
            
            while wait_for_action_finished == True:
                try:
                    wait_for_action_finished = False
                    #state is of form [pos X, pos Y, is Active, ball colour] x number of balls + player colour
                    state, reward, terminal = receive_response(s)
                    #The terminal is somehow inverted from what is being sent from unity, idk why
                    
                    print(f"Received response: state={state}, reward={reward}, terminal={terminal}")
                except:
                    #brain tired, how can i have nothing here?
                    wait_for_action_finished = True
                    x = 1

if __name__ == '__main__':
    main()