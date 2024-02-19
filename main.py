import streamlit as st
import scraper
import utils
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

### INITIALIZATION ###
if "load_state" not in st.session_state:
     st.session_state.load_state = False
if "rating_round" not in st.session_state:
     st.session_state.rating_round = 'Continuous (0 to 1)'
if "plot_type" not in st.session_state:
     st.session_state.plot_type = 'Line'

twitter_model = utils.download_model(utils.MODEL_NAME_TWITTER)
amazon_model = utils.download_model(utils.MODEL_NAME_AMAZON)

utils.init()


### MAIN APP ###
st.title('Review Scraper')
st.caption('Scrapes [Google](https://www.google.com/) for reviews of a product')
st.caption('Open source on [GitHub link](https://github.com/leezhongjun/ReviewScraper)')
st.caption('Powered by [HuggingFace](https://huggingface.co/), [Streamlit](https://streamlit.io/), and [Apify](https://apify.com/)')

st.markdown("""
    <style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 17px;

}
.big-font {
    font-size: 18px;
}
    </style>
    """, unsafe_allow_html=True)

st.warning('Apify has a limit of 50,000 queries a month', icon="⚠️")
col1, col2 = st.columns(2)
with col1:
    query_usage = st.button('See current usage')
if query_usage:
    with col2:
        st.write(scraper.query_for_usage())


form = st.form(key='main_form')
item = form.text_input('Enter an item (product, movie, etc.) to search for reviews: ', value='PS5 Console')
num_of_queries = form.slider('Number of queries: ', min_value=100, max_value=1000, value=100, step=100)
submit = form.form_submit_button('Search')
if submit or st.session_state.load_state:
    st.session_state.load_state = True
    st.write(f'Searching for reviews for **{item}**...')
    desc_dataset, rating_dataset = scraper.query_google(item, num_of_queries, use_json=False) # use_json=True for dummy data
    st.write(f'<span class=big-font>Found `{len(desc_dataset)}` reviews</span>', unsafe_allow_html = True)
    st.write(f'<span class=big-font>Found `{len(rating_dataset)}` ratings</span>', unsafe_allow_html = True)

    if len(rating_dataset) > 0:
        st.markdown("***")
        st.subheader('Raw ratings')

        col1, col2 = st.columns(2)
        with col1:
            rating_round_d = {'Continuous (0 to 1)': 0, 'Out of 5': 5, 'Out of 10': 10}
            st.session_state.rating_round = st.radio('Rating precision:', rating_round_d.keys(), index=0)

        with col2:
            plot_types_ls = ['Line', 'Scatter', 'Both']
            st.session_state.plot_type = st.radio('Chart type (for continuous only):', plot_types_ls, index=2)

        fig = scraper.show_ratings(rating_dataset, rating_round=rating_round_d[st.session_state.rating_round], plot_type=st.session_state.plot_type)
        st.pyplot(fig)
        multiplier = 1 if st.session_state.rating_round == 'Continuous (0 to 1)' else rating_round_d[st.session_state.rating_round]
        st.write(f'<span class=big-font><b>Average <u>raw</u> rating:</b> `{np.mean(rating_dataset) * multiplier: .2f} / {multiplier}`</span>', unsafe_allow_html = True)

    if len(desc_dataset) > 0:
        st.markdown("***")
        st.subheader('Sentiment Analysis')
        col1, col2 = st.columns(2)
        with col1:
            fig, mean = scraper.eval_sentiment(desc_dataset, amazon_model, utils.AMAZON_ROW_LABELS, utils.AMAZON_ROW_VALUES, 'Rating from 1 - 5 stars')
            st.pyplot(fig)
            st.write(f'<span class=big-font><b>Average rating from <u>1 - 5</u>:</b> `{mean: .2f} / 5`</span>', unsafe_allow_html = True)
        with col2:
            fig, mean = scraper.eval_sentiment(desc_dataset, twitter_model, utils.TWITTER_ROW_LABELS, utils.TWITTER_ROW_VALUES, 'Negative / Neutral / Positive')
            st.pyplot(fig)
            st.write(f'<span class=big-font><b>Average rating from <u>0 / 0.5 / 1</u>:</b> `{mean: .2f} / 1`</span>', unsafe_allow_html = True)

        st.markdown("***")
        wc_figures = scraper.create_wordcloud(desc_dataset)
        st.subheader('Word clouds')
        col3, col4, col5 = st.columns(3)
        with col3:
            st.pyplot(wc_figures[0])
        with col4:
            st.pyplot(wc_figures[1])
        with col5:
            st.pyplot(wc_figures[2])

    


