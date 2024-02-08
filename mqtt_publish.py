import paho.mqtt.client as mqtt
import time

def publish_top_configuration(config_log):
    # Replace with your information
    broker_address = "broker.hivemq.com"  # e.g., "192.168.1.10" or "broker.example.com"
    broker_port = 1883  # Replace with your broker's port
    topic = "emptySeats/HardwareToApp"  # Replace with your topic
    username = "leon"  # Replace if your broker requires authentication
    password = "leon"  # Replace if your broker requires authentication

    def on_connect(client, userdata, flags, rc):
        print(f"Connected with result code {rc}")

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
        # Publish the most frequent configuration
        client.publish(topic, top_config.replace(' ', ''))

        # Wait for a while to ensure the message is sent
        time.sleep(2)

    # Stop the loop and disconnect
    client.loop_stop()
