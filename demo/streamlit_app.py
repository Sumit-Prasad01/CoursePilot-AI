import streamlit as st
from dotenv import load_dotenv

from src.course_planner_agent.graph.workflow import CoursePlannerWorkflow

load_dotenv()

st.set_page_config(page_title="CoursePilot AI", layout="wide")

st.title("🎓 CoursePilot AI")
st.markdown("AI-powered Course Planning Assistant (RAG + LangGraph)")

# Initialize workflow
if "workflow" not in st.session_state:
    st.session_state.workflow = CoursePlannerWorkflow()

query = st.text_area("Enter your query:", height=120)

if st.button("Run"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = st.session_state.workflow.run(query)

        st.subheader("📌 Answer")
        st.write(result["final_output"]["answer"])

        if result.get("citations"):
            st.subheader("📚 Citations")
            st.write(result["citations"])

        if result.get("error"):
            st.error(result["error"])