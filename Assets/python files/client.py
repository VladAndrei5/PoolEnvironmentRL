import socket

def check_wait(s):
    waiting = True
    while waiting: 
        data = s.recv(1024).decode()
        if data.startswith("WAIT"):
            waiting = True
        else:
            waiting = False 
            print("stop waiting")

def receive_state(s):
    check_wait(s)
    data = s.recv(1024).decode()
    parts = data.split(',')
    state = [float(x) for x in parts[:-2]]
    reward = int(parts[-2])
    terminal = bool(parts[-1])
    # for some reason the terminal is reversed
    if terminal == True:
        terminal = False
    else:
        terminal = True
    s.sendall("RECIEVEDSTATE".encode())    
    print(f"Received response: state={state}, reward={reward}, terminal={terminal}")
    return state, reward, terminal       

def send_instruction(s, instruction):
    print("sending instruction")
    check_wait(s)
    #s.sendall(f"{instruction[0]},{instruction[1]}".encode())
    s.sendall(f"INSTRUCTION,{instruction[0]},{instruction[1]}".encode())
    print(f"Sent instruction: {instruction}")
    
def reset_env(s):
    check_wait(s)
    s.sendall("RESET".encode())
    print("Sent Reset command")

def main():
    host = '127.0.0.1'
    port = 8888
    
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print("Connected to the server.")
        
        reset_env(s)
        state, reward, terminal = receive_state(s)
        
        for i in range(9):  # Repeat the cycle 5 times
            
            if i == 2:
                reset_env(s)
                state, reward, terminal = receive_state(s)
    
            else:
                instruction = (130, 0.3)  # Replace with your actual instruction values
                send_instruction(s, instruction)
                state, reward, terminal = receive_state(s)
                



if __name__ == '__main__':
    main()