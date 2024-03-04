import paho.mqtt.client as mqtt
import time

# Replace with your information
broker_address = "broker.hivemq.com"
broker_port = 1883 
topic = "emptySeats/HardwareToApp"
topic_screenshot = "emptySeats/Screenshot" 

def publish_top_configuration(config_log):
    
    def on_connect(client, userdata, flags, rc):
        print(f"Publisher Connected with result code {rc}")

    def on_publish(client, userdata, mid):
        print(f"Message Published: {mid}")
        print(top_config)

    # Create MQTT client instance
    client = mqtt.Client()

    # Set callback functions
    client.on_connect = on_connect
    client.on_publish = on_publish

    # Connect to the broker
    client.connect(broker_address, broker_port, 60)

    # Start the loop
    client.loop_start()

    # Find the most frequent configuration
    max_frequency = 0
    top_config = None
    for config, frequency in config_log.items():
        if frequency > max_frequency:
            max_frequency = frequency
            top_config = config

    if top_config is not None:
        

        # Publish a screenshot of the most frequent configuration
        with open("./screenshot.jpg",'rb') as file:
            filecontent = file.read()
            byteArr = bytearray(filecontent)
            client.publish(topic_screenshot,byteArr,2)
            print("Screenshot sent to the app")

            # Publish the most frequent configuration
            client.publish(topic, top_config.replace(' ', ''))

        # msg_status = result[0]
        # if msg_status == 0:
        #    print(f"message sent to topic {topic_screenshot}")
        # else:
        #    print(f"Failed to send message to topic {topic_screenshot}")

        # Wait for a while to ensure the message is sent
        time.sleep(3)

    # Stop the loop and disconnect
    client.loop_stop()

    	