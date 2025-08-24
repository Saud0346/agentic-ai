import streamlit as st
import resume-parser

st.set_page_config(page_title="💬 Resume Chatbot", layout="wide")
st.title("💬 Resume Chatbot")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf","docx","txt","csv","xlsx","xls","json","html"])

if uploaded_file:
    file_path = "temp_resume_" + uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Read file
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = resume-parser.read_any_file(file_path)

    # Step 2: Initial analyze for tick/cross display
    st.session_state.processed_data = resume-parser.analyze_and_validate_resume(st.session_state.raw_data)
    st.session_state.cv_json = None

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(st.session_state.processed_data)

    # Step 3: User input loop
    while True:
        user_input = st.chat_input("Enter missing information (or leave empty to skip)")

        if user_input:
            # Append new info to RAW data
            st.session_state.raw_data += f"\n\n{user_input}"

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Re-analyze RAW data for updated tick/cross
            st.session_state.processed_data = resume-parser.analyze_and_validate_resume(st.session_state.raw_data)

            # Show updated tick/cross
            with st.chat_message("assistant"):
                st.markdown(st.session_state.processed_data)

            # Check if all categories are ✅
            if all("✅" in line for line in st.session_state.processed_data.splitlines() if line.strip().startswith(("-", "*"))):
                # Generate JSON from RAW data
                st.session_state.cv_json = resume-parser.get_clean_resume_json(st.session_state.raw_data)
                if st.session_state.cv_json:
                    with st.expander("📝 Structured JSON"):
                        st.json(st.session_state.cv_json)
                break  # All done
        else:
            break