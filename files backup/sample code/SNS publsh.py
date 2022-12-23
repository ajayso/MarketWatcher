# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:10:43 2021

@author: Ajay Solanki
"""
import json
import logging
import time
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SNS_Wrapper:
    
    def __init__(self,sns_resource):
        self.sns_resource = sns_resource
    
    def create_topic(self, name):
        try:
            topic = self.sns_resource.create_topic(Name=name)
            logger.info("Created topic %s with ARN %s.", name, topic.arn)
        except:
            logger.exception("Couldn't create topic %s.", name)
            raise
        else:
            return topic
    
    def list_topics(self):
        """
        Lists topics for the current account.

        :return: An iterator that yields the topics.
        """
        try:
            topics_iter = self.sns_resource.topics.all()
            logger.info("Got topics.")
        except ClientError:
            logger.exception("Couldn't get topics.")
            raise
        else:
            return topics_iter
    
    @staticmethod
    def delete_topic(topic):
        """
        Deletes a topic. All subscriptions to the topic are also deleted.
        """
        try:
            topic.delete()
            logger.info("Deleted topic %s.", topic.arn)
        except ClientError:
            logger.exception("Couldn't delete topic %s.", topic.arn)
            raise
    
    @staticmethod
    def subscribe(topic, protocol, endpoint):
        try:
            subscription = topic.subscribe(
                Protocol=protocol, Endpoint=endpoint, ReturnSubscriptionArn=True)
            logger.info("Subscribed %s %s to topic %s.", protocol, endpoint, topic.arn)
        except ClientError:
            logger.exception(
                "Couldn't subscribe %s %s to topic %s.", protocol, endpoint, topic.arn)
            raise
        else:
            return subscription
    
    def list_subscriptions(self, topic=None):
        """
        Lists subscriptions for the current account, optionally limited to a
        specific topic.

        :param topic: When specified, only subscriptions to this topic are returned.
        :return: An iterator that yields the subscriptions.
        """
        try:
            if topic is None:
                subs_iter = self.sns_resource.subscriptions.all()
            else:
                subs_iter = topic.subscriptions.all()
            logger.info("Got subscriptions.")
        except ClientError:
            logger.exception("Couldn't get subscriptions.")
            raise
        else:
            return subs_iter
    
    @staticmethod
    def add_subscription_filter(subscription, attributes):
        try:
           att_policy = {key: [value] for key, value in attributes.items()}
           subscription.set_attributes(
               AttributeName='FilterPolicy', AttributeValue=json.dumps(att_policy))
           logger.info("Added filter to subscription %s.", subscription.arn)
        except ClientError:
           logger.exception(
               "Couldn't add filter to subscription %s.", subscription.arn)
           raise
    @staticmethod
    def delete_subscription(subscription):
        """
        Unsubscribes and deletes a subscription.
        """
        try:
            subscription.delete()
            logger.info("Deleted subscription %s.", subscription.arn)
        except ClientError:
            logger.exception("Couldn't delete subscription %s.", subscription.arn)
            raise
    
    
    def publish_text_message(self, phone_number, message):
        try:
            response = self.sns_resource.meta.client.publish(
                PhoneNumber=phone_number, Message=message)
            message_id = response['MessageId']
            logger.info("Published message to %s.", phone_number)
        except ClientError:
            logger.exception("Couldn't publish message to %s.", phone_number)
            raise
        else:
            return message_id
    
    @staticmethod
    def publish_message(topic, message, attributes):
        """
        Publishes a message, with attributes, to a topic. Subscriptions can be filtered
        based on message attributes so that a subscription receives messages only
        when specified attributes are present.

        :param topic: The topic to publish to.
        :param message: The message to publish.
        :param attributes: The key-value attributes to attach to the message. Values
                           must be either `str` or `bytes`.
        :return: The ID of the message.
        """
        try:
            att_dict = {}
            for key, value in attributes.items():
                if isinstance(value, str):
                    att_dict[key] = {'DataType': 'String', 'StringValue': value}
                elif isinstance(value, bytes):
                    att_dict[key] = {'DataType': 'Binary', 'BinaryValue': value}
            response = topic.publish(Message=message, MessageAttributes=att_dict)
            message_id = response['MessageId']
            logger.info(
                "Published message with attributes %s to topic %s.", attributes,
                topic.arn)
        except ClientError:
            logger.exception("Couldn't publish message to topic %s.", topic.arn)
            raise
        else:
            return message_id
    
    @staticmethod
    def publish_multi_message(
            topic, subject, default_message, sms_message, email_message):
        """
        Publishes a multi-format message to a topic. A multi-format message takes
        different forms based on the protocol of the subscriber. For example,
        an SMS subscriber might receive a short, text-only version of the message
        while an email subscriber could receive an HTML version of the message.

        :param topic: The topic to publish to.
        :param subject: The subject of the message.
        :param default_message: The default version of the message. This version is
                                sent to subscribers that have protocols that are not
                                otherwise specified in the structured message.
        :param sms_message: The version of the message sent to SMS subscribers.
        :param email_message: The version of the message sent to email subscribers.
        :return: The ID of the message.
        """
        try:
            message = {
                'default': default_message,
                'sms': sms_message,
                'email': email_message
            }
            response = topic.publish(
                Message=json.dumps(message), Subject=subject, MessageStructure='json')
            message_id = response['MessageId']
            logger.info("Published multi-format message to topic %s.", topic.arn)
        except ClientError:
            logger.exception("Couldn't publish message to topic %s.", topic.arn)
            raise
        else:
            return message_id

# test code
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sns_wrapper= SNS_Wrapper(boto3.resource('sns'))
topic_name = f'fx_demo-{time.time_ns()}'

topic = sns_wrapper.create_topic(topic_name)
phone_number = input(
        "Enter a phone number (in E.164 format) that can receive SMS messages: ")
if phone_number != '':
    print(f"Sending an SMS message directly from SNS to {phone_number}.")
    sns_wrapper.publish_text_message(phone_number, 'Your number is tracked by Dropoutjeep ')

email = input(
        f"Enter an email address to subscribe to {topic_name} and receive "
        f"a message: ")

if email != '':
    print(f"Subscribing {email} to {topic_name}.")
    email_sub = sns_wrapper.subscribe(topic, 'email', email)
    answer = input(
        f"Confirmation email sent to {email}. To receive SNS messages, "
        f"follow the instructions in the email. When confirmed, press "
        f"Enter to continue.")
    while (email_sub.attributes['PendingConfirmation'] == 'true'
           and answer.lower() != 's'):
        answer = input(
            f"Email address {email} is not confirmed. Follow the "
            f"instructions in the email to confirm and receive SNS messages. "
            f"Press Enter when confirmed or enter 's' to skip. ")
        email_sub.reload()
phone_sub = None
if phone_number != '':
    print(f"Subscribing {phone_number} to {topic_name}. Phone numbers do not "
          f"require confirmation.")
    phone_sub = sns_wrapper.subscribe(topic, 'sms', phone_number)

if phone_number != '' or email != '':
    print(f"Publishing a multi-format message to {topic_name}. Multi-format "
          f"messages contain different messages for different kinds of endpoints.")
    for number in range(2):
        sns_wrapper.publish_multi_message(
            topic, 'SNS multi-format demo',
            'We have enabled tracking on your phone.',
            'You will be monitored by Dropoutjeep .',
            'This is an informational message no action required')        
if phone_sub is not None:
    mobile_key = 'mobile'
    friendly = 'friendly'
    print(f"Adding a filter policy to the {phone_number} subscription to send "
          f"only messages with a '{mobile_key}' attribute of '{friendly}'.")
    sns_wrapper.add_subscription_filter(phone_sub, {mobile_key: friendly})
    print(f"Publishing a message with a {mobile_key}: {friendly} attribute.")
    sns_wrapper.publish_message(
        topic, "Hello! This message is mobile friendly.", {mobile_key: friendly})
    not_friendly = 'not-friendly'
    print(f"Publishing a message with a {mobile_key}: {not_friendly} attribute.")
    sns_wrapper.publish_message(
        topic,
        "Hey. This message is not mobile friendly, so you shouldn't get "
        "it on your phone.",
        {mobile_key: not_friendly})

print(f"Getting subscriptions to {topic_name}.")
topic_subs = sns_wrapper.list_subscriptions(topic)
for sub in topic_subs:
    print(f"{sub.arn}")

print(f"Deleting subscriptions and {topic_name}.")
for sub in topic_subs:
    if sub.arn != 'PendingConfirmation':
        sns_wrapper.delete_subscription(sub)
sns_wrapper.delete_topic(topic)

print("Thanks for watching!")