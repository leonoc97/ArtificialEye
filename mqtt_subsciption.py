import paho.mqtt.client as mqtt

# Replace with your MQTT broker's information
broker_address = "broker.hivemq.com"
broker_port = 1883
topic = "emptySeats/AppToHardware"


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # Subscribe to the topic when connected
    client.subscribe(topic)


def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()}")
    if message.payload.decode == "start":
        print("1")

# Create an MQTT client instance
client = mqtt.Client()

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, broker_port, 60)

# Start the loop
client.loop_forever()
