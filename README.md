# Whatsapp
import streamlit as st
import numpy as np
import seaborn as sn
import pandas as pd
import re


def gettimeanddate(string):
    string = string.split(',')
    date, time = string[0], string[1]
    time = time.split('-')
    time = time[0].strip()

    return date+" "+time


def getstring(text):
    return text.split('\n')[0]


def preprocess(data):

    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_messages': messages,
                       'message_date': dates})

    df['message_date'] = df['message_date'].apply(
        lambda text: gettimeanddate(text))
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []

    for message in df['user_messages']:

        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])

        else:
            users.append('Group Notification')
            messages.append(entry[0])

    df['User'] = users
    df['message'] = messages

    df['message'] = df['message'].apply(lambda text: getstring(text))

    df = df.drop(['user_messages'], axis=1)
    df = df[['message', 'date', 'User']]

    df = df.rename(columns={'message': 'Message',
                            'date': 'Date'})

    df['Only date'] = pd.to_datetime(df['Date']).dt.date

    df['Year'] = pd.to_datetime(df['Date']).dt.year

    df['Month_num'] = pd.to_datetime(df['Date']).dt.month

    df['Month'] = pd.to_datetime(df['Date']).dt.month_name()

    df['Day'] = pd.to_datetime(df['Date']).dt.day

    df['Day_name'] = pd.to_datetime(df['Date']).dt.day_name()

    df['Hour'] = pd.to_datetime(df['Date']).dt.hour

    df['Minute'] = pd.to_datetime(df['Date']).dt.minute

    return df
import streamlit as st
import preprocess
import re
import stats
import matplotlib.pyplot as plt
import numpy as np


st.sidebar.title("Whatsapp Chat Analyzer")

# this is for uploading a file

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # converting the bytecode to the text-file

    data = bytes_data.decode("utf-8")

    # sending the file data to the preprocess function for further functioning

    df = preprocess.preprocess(data)

    # displaying the dataframe

    # st.dataframe(df)

    # fetch unique users
    user_list = df['User'].unique().tolist()

    # removing the groupnotification

    user_list.remove('Group Notification')

    # organinsing things
    user_list.sort()

    # including overall,this will be responsible for showcasing the  overall chat group analysis

    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "Show analysis with respect to", user_list)

    st.title("Whats App Chat Analysis for " + selected_user)
    if st.sidebar.button("Show Analysis"):

        # getting the stats of the selected user from the stats script

        num_messages, num_words, media_omitted, links = stats.fetchstats(
            selected_user, df)

        # first phase is to showcase the basic stats like number of users,number of messages,number of media shared and all,so for that i requrire the 4 columns

        col1, col2, col3, col4 = st.beta_columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total No.of Words")
            st.title(num_words)

        with col3:
            st.header("Media Shared")
            st.title(media_omitted)

        with col4:
            st.header("Total Links Shared")
            st.title(links)

        # finding the busiest users in the group

        if selected_user == 'Overall':

            # dividing the space into two columns
            # first col is the bar chart and the second col is the dataframe representing the

            st.title('Most Busy Users')
            busycount, newdf = stats.fetchbusyuser(df)
            fig, ax = plt.subplots()
            col1, col2 = st.beta_columns(2)
            with col1:
                ax.bar(busycount.index, busycount.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(newdf)

        # Word Cloud

        st.title('Word Cloud')
        df_img = stats.createwordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_img)
        st.pyplot(fig)

        # most common words in the chat

        most_common_df = stats.getcommonwords(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most commmon words')
        st.pyplot(fig)

        # Emoji Analysis

        emoji_df = stats.getemojistats(selected_user, df)
        emoji_df.columns = ['Emoji', 'Count']

        st.title("Emoji Analysis")

        col1, col2 = st.beta_columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            emojicount = list(emoji_df['Count'])
            perlist = [(i/sum(emojicount))*100 for i in emojicount]
            emoji_df['Percentage use'] = np.array(perlist)
            st.dataframe(emoji_df)

        # Monthly timeline

        st.title("Monthly Timeline")
        time = stats.monthtimeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(time['Time'], time['Message'], color='green')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        st.pyplot(fig)

        # Activity maps

        st.title("Activity Maps")

        col1, col2 = st.beta_columns(2)

        with col1:

            st.header("Most Busy Day")

            busy_day = stats.weekactivitymap(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:

            st.header("Most Busy Month")
            busy_month = stats.monthactivitymap(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)

