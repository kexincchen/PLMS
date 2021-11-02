using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if !UNITY_EDITOR
using System.Threading.Tasks;
using Windows.Networking;
using Windows.Networking.Sockets;
using System;
using Windows.Storage.Streams;
#endif


public class UDPSocket 
{

#if !UNITY_EDITOR
        DatagramSocket socket;
        HostName remoteIP;
        string remotePort;

        HostName listenIP;
        string listenPort;

        Queue<byte[]> messageQueue;

        public bool MsgAvailable()
        {
            return this.messageQueue.Count>0?true:false;
        }

        public UDPSocket(NetworkSettings networkSettings)
        {
            this.remoteIP = new HostName(networkSettings.remoteIP);
            this.remotePort = networkSettings.remotePort;

            this.listenIP = new HostName(networkSettings.listenIP);
            this.listenPort = networkSettings.listenPort;

            this.socket = new DatagramSocket();
            this.messageQueue = new Queue<byte[]>();
        }

        public async void Listen()
        {
            this.socket.MessageReceived+=this.Socket_MessageReceived;

            try
            {
                //await this.socket.BindEndpointAsync(this.listenIP, this.listenPort);
                await this.socket.BindServiceNameAsync(this.listenPort);

            }catch(Exception e)
            {
                Debug.Log("ERROR: "+e.ToString());
            }
            
            Debug.Log("Listening on ip:"+this.listenIP+" and port:"+this.listenPort);
        }

        private void Socket_MessageReceived(DatagramSocket sender, DatagramSocketMessageReceivedEventArgs args)
        {
            //Debug.Log("Received Message");

            DataReader dataReader = args.GetDataReader();

            Byte[] buffer = new Byte[dataReader.UnconsumedBufferLength];
            dataReader.ReadBytes(buffer);

            this.messageQueue.Enqueue(buffer);
        }

        public async void SendMessage(byte[] message)
        {
            IOutputStream streamOut = await this.socket.GetOutputStreamAsync(this.remoteIP, this.remotePort);
            DataWriter dataWriter = new DataWriter(streamOut);
            dataWriter.WriteBytes(message);
            await dataWriter.StoreAsync();
        }

        public byte[] ReceiveMsg()
        {
            return this.messageQueue.Dequeue();
        }
#endif


}
