import json
import jsonpickle
import msgpack
import socket
import sys,getopt
import threading

from Entity.testEntity import TestEntity
from Entity.jsonMessage import JsonMessage

#for listening
def bindSock(sock,ipv4,port):
    sock.bind((ipv4,port))

def listen(sock):
    while True:
        data,addr = sock.recvfrom(1024)
        print "received message"
        #handleMsg(data)
        handleMsgJson(data)

#decode msgpack msg
def handleMsg(data):
    msg = msgpack.loads(data)
    print msg

def handleMsgJson(data):
    msg = deserializeMsgJson(data)
    jm = JsonMessage.from_json(msg)
    print "type = " + jm.messageType

    if(jm.messageType == "PostItContent"):
        print str(jm.messageObject.id)
        print jm.messageObject.clue
        print jm.messageObject.header
        print str(jm.messageObject.topics)

def initSocket():
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    return sock

## message here is always a jsonMessage.JsonMessage
def sendUDPMsg(sock, msg, ipv4, port):
    serializeMsg = serializeMsgJson(msg)
    sock.sendto(serializedMsg,(ipv4,port))

def serializeMsg(msg,encoder = None):
    if encoder != None:
        return msgpack.packb(msg, default=encoder)
    return msgpack.packb(msg);

def serializeMsgJson(msg):
    return jsonpickle.encode(msg)

def deserializeMsgJson(msg):
    return jsonpickle.decode(msg)

def main(argv):
    udp_ip_remote = "192.168.137.183"
    udp_port_remote = 12344

    udp_ip_listen = "0.0.0.0"
    udp_port_listen = 12344

    try:
        opts,args = getopt.getopt(argv,"hli:p:",["ip=","port="])
    except getopt.GetoptError:
        print "the ip address and port are for the remote receiver\n"
        print "udp_test.py -i <ipaddress> -p <port number>\n"
        print "udp_test.py --ip=<ipaddress> --port=<port number>\n"
        print "upd_test.py -h"
        print "for help\n"
        print "udp_test -l"
        print "for localhost with port=12344\n"
        sys.exit(2)

    #exit if no arguments
    if(not opts):
        print "the ip address and port are for the remote receiver\n"
        print "udp_test.py -i <ipaddress> -p <port number>\n"
        print "udp_test.py --ip=<ipaddress> --port=<port number>\n"
        print "upd_test.py -h"
        print "for help\n"
        print "udp_test -l"
        print "for localhost with port=12344\n"
        sys.exit(2)


    for opt,val in opts:
        if(opt == "-h"):
            print "the ip address and port are for the remote receiver\n"
            print "udp_test.py -i <ipaddress> -p <port number>\n"
            print "udp_test.py --ip=<ipaddress> --port=<port number>\n"
            print "upd_test.py -h"
            print "for help\n"
            print "udp_test -l"
            print "for localhost with port=12344\n"
            sys.exit(2)
        if(opt == "-l"):
            break
        if(opt in ("-i","--ip")):
            udp_ip_remote = val
        if(opt in ("-p","--port")):
            udp_port_remote = int(val)


    #open listen socket
    listenSock = initSocket()
    bindSock(listenSock, udp_ip_listen, udp_port_listen)
    listen(listenSock)

    #maybe I don't need to run this on a separate thread
    #listen_UDP = threading.Thread(target = listen, args = (listenSock,))
    #listen_UDP.start()

if __name__ == '__main__':
    main(sys.argv[1:])
