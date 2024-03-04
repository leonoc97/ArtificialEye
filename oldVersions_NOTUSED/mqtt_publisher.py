import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

def publish_top_configuration(config_log):
    # MQTT broker settings
    broker_address = "broker.hivemq.com"
    broker_port = 1883
    topic = "emptySeats/HardwareToApp"

    # Find the most frequent configuration
    max_frequency = 0
    top_config = None
    for config, frequency in config_log.items():
        if frequency > max_frequency:
            max_frequency = frequency
            top_config = config

    if top_config is not None:
        print(f"Most frequent configuration: {top_config}, Frequency: {max_frequency}")

        # Create an MQTT client instance
        client = mqtt.Client()

        # Connect to the MQTT broker
        client.connect(broker_address, broker_port)
        if client.is_connected():
            print("Connected to MQTT broker successfully.")

        # Publish the most frequent configuration
        payload = f"Configuration: {top_config.replace(' ', '')}"
        client.publish(topic, payload)

        # Disconnect from the MQTT broker
        client.disconnect()
        print("Published the most frequent configuration.")
        print(f"Published message: Configuration: {top_config.replace(' ', '')}, Frequency: {max_frequency}")

    else:
        print("No configurations found in the log.")

if __name__ == "__main__":
    # Sample config_log dictionary
    config_log = {
        "[1, 0, 0, 1, 0, 1]": 10,
        "[0, 1, 0, 0, 0, 0]": 7,
        "[0, 0, 1, 0, 0, 0]": 5
    }

    # Publish the most frequent configuration
    publish_top_configuration(config_log)
    publish.single("EmptySeats/HardwareToApp", "Hello", hostname="broker.hivemq.com")

