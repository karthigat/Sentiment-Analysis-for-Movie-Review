# if idx < len(Image):
#     cols[3].image(Image[idx], width=150, caption=list_movie[idx])
#     idx = idx + 1

# LOGO_IMAGE = "Logo.jpg"
# for k, t in zip(Image, list_movie):
#     # st.markdown(i)
#     col1,col2 = st.beta_columns([3,3])
#     with col1:
#         st.markdown(
#             """
#             <style>
#             .container {
#                 display: flex;
#             }
#             .logo-text {
#                 font-weight:700 !important;
#                 font-size:50px !important;
#                 color: #f9a01b !important;
#                 padding-top: 75px !important;
#             }
#             .logo-img {
#                 float:right;
#             }
#             .grid-container{
#                 display: grid
#             }
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#
#         st.markdown(
#             f"""
#             <div class="container">
#                 <img width="150" height="150" class="logo-img" src="data:image/png;base64,{base64.b64encode(open(k, "rb").read()).decode()}">
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#
#     with col2:
#         st.markdown(
#         f"""<p>{t}</p>
#         """,
#         unsafe_allow_html=True
#         )



# idx = 0
    # while idx < len(Image):
    #     for _ in range(len(Image)):
    #         cols = st.beta_columns(3)
    #
    #         # first, going to loop over the columns,
    #         for col_num in range(3):
    #
    #             # next check that idx is in range, if it is then we add an image to the
    #             # column number we are on. If idx > len(filteredImages), it should
    #             # skip those columns, they will exist but we just wont put
    #             # anything in them
    #             if idx <= len(Image):
    #                 cols[col_num].image(Image[idx],
    #                                     width=150, caption=movies[idx])
    #
    #                 idx += 1