import socket

def check_wait(s):
    waiting = True
    while waiting: 
        data = s.recv(1024).decode()
        if data.startswith("WAIT"):
            waiting = True
        else:
            waiting = False 

def receive_state(s):
    check_wait(s)
    data = s.recv(1024).decode()
    parts = data.split(',')
    state = [float(x) for x in parts[:-2]]
    reward = int(parts[-2])
    terminal = (parts[-1])
    if terminal == "False":
        terminal = False
    else:
        terminal = True
    # for some reason the terminal is reversed
    s.sendall("RECIEVEDSTATE".encode())    
    #print(f"Received response: state={state}, reward={reward}, terminal={terminal}")
    return state, reward, terminal       

def send_instruction(s, instruction):
    check_wait(s)
    s.sendall(f"INSTRUCTION,{instruction[0]},{instruction[1]},{instruction[2]}".encode())
    #print(f"Sent instruction: {instruction}")
    
def reset_env(s):
    check_wait(s)
    s.sendall("RESET".encode())
    #print("Sent Reset command")

def main():
    host = '127.0.0.1'
    port = 8888
    
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("Connected to the server.")
        
        
        for i in range(900):  # Repeat the cycle 5 times
            
            if i == 5:
                reset_env(s)
                state, reward, terminal = receive_state(s)
                if terminal == True:
                    reset_env(s)
                    state, reward, terminal = receive_state(s)
    
            else:
                instruction = (5, 0, 1)  # Replace with your actual instruction values
                print(instruction)
                send_instruction(s, instruction)
                state, reward, terminal = receive_state(s)
                if terminal == True:
                    reset_env(s)
                    state, reward, terminal = receive_state(s)
                



if __name__ == '__main__':
    main()
