import streamlit as st
import pandas as pd
import pdfplumber
import re

st.title("GRC Panel Specification Extractor")

uploaded_file = st.file_uploader("Upload a PDF, Excel, or CSV file", type=["pdf", "xlsx", "xls", "csv"])

def extract_from_pdf(file):
    data = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            matches = re.findall(r'(Grc\.[\w\.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', text)
            for match in matches:
                unit_type, count, height, width, depth = match
                data.append({
                    "Type": unit_type,
                    "Count": int(count),
                    "Height": int(height),
                    "Width": int(width),
                    "Depth": int(depth),
                    "Weight": None  # Optional column
                })
    return pd.DataFrame(data)


def extract_from_excel_or_csv(file):
    # Extracts and renames columns based on the selected header row
    # Reverted: Load entire file from top
    try:
            panel_rows = []
            for _, row in df.iterrows():
                count = int(row['Count']) if 'Count' in row and pd.notna(row['Count']) else 1
                for _ in range(count):
                    panel_rows.append({
                        'Type': row['Type'],
                        'Height': float(row['Height']),
                        'Width': float(row['Width']),
                        'Depth': float(row['Depth']),
                        'Weight': float(row['Weight']) if 'Weight' in row and pd.notna(row['Weight']) else float(row['Height']) * float(row['Width']) * (panel_thickness / 1000) * (density / 1000)
                    })

            beds, trucks = compute_beds_and_trucks(panel_rows, bed_width, bed_weight_limit, truck_weight_limit, truck_max_length)
            st.subheader("Transport Packing Plan")

            # Validation warnings
            overfilled_beds = [i for i, bed in enumerate(beds) if bed['Width'] > bed_width or bed['Weight'] > bed_weight_limit]
            if overfilled_beds:
                st.warning(f"⚠️ Overfilled Beds: {overfilled_beds}")

            overfilled_trucks = [i for i, truck in enumerate(trucks) if sum(b['Length'] for b in truck) > truck_max_length or sum(b['Weight'] for b in truck) > truck_weight_limit]
            if overfilled_trucks:
                
            st.warning(f"⚠️ Overfilled Trucks: {overfilled_trucks}")
        try:
            panel_rows = []
            for _, row in df.iterrows():
                count = int(row['Count']) if 'Count' in row and pd.notna(row['Count']) else 1
                for _ in range(count):
                    panel_rows.append({
                        'Type': row['Type'],
                        'Height': float(row['Height']),
                        'Width': float(row['Width']),
                        'Depth': float(row['Depth']),
                        'Weight': float(row['Weight']) if 'Weight' in row and pd.notna(row['Weight']) else float(row['Height']) * float(row['Width']) * (panel_thickness / 1000) * (density / 1000)
                    })

            beds, trucks = compute_beds_and_trucks(panel_rows, bed_width, bed_weight_limit, truck_weight_limit, truck_max_length)
            st.markdown(f"**Total Beds:** {len(beds)}")
            st.markdown(f"**Total Trucks:** {len(trucks)}")

            # Bed usage visualization
            import matplotlib.pyplot as plt

            usage_data = pd.DataFrame(beds)
            usage_data['Used Width (mm)'] = usage_data['Width']
            usage_data['Used Weight (kg)'] = usage_data['Weight']

            fig1, ax1 = plt.subplots()
            ax1.bar(range(len(beds)), usage_data['Used Width (mm)'])
            ax1.set_title("Used Bed Width per Bed")
            ax1.set_xlabel("Bed Index")
            ax1.set_ylabel("Width (mm)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.bar(range(len(beds)), usage_data['Used Weight (kg)'])
            ax2.set_title("Used Bed Weight per Bed")
            ax2.set_xlabel("Bed Index")
            ax2.set_ylabel("Weight (kg)")
            st.pyplot(fig2)

            # Truck usage visualization
            truck_weights = [sum(b['Weight'] for b in truck) for truck in trucks]
            truck_lengths = [sum(b['Length'] for b in truck) for truck in trucks]

            fig3, ax3 = plt.subplots()
            ax3.bar(range(len(trucks)), truck_weights)
            ax3.set_title("Total Weight per Truck")
            ax3.set_xlabel("Truck Index")
            ax3.set_ylabel("Weight (kg)")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots()
            ax4.bar(range(len(trucks)), truck_lengths)
            ax4.set_title("Used Length per Truck")
            ax4.set_xlabel("Truck Index")
            ax4.set_ylabel("Length (mm)")
            st.pyplot(fig4)

            for i, truck in enumerate(trucks):
                st.markdown(f"### Truck {i+1} - Beds: {len(truck)}")
                truck_df = pd.DataFrame(truck)
                st.dataframe(truck_df)

            export_data = []
            for i, truck in enumerate(trucks):
                for j, bed in enumerate(truck):
                    export_data.append({
                        'Truck': i + 1,
                        'Bed': j + 1,
                        'Length': bed['Length'],
                        'Height': bed['Height'],
                        'Width': bed['Width'],
                        'Weight': bed['Weight'],
                        'Num Panels': bed['Num Panels'],
                        'Panel Types': ', '.join(bed['Panel Types'])
                    })

            export_df = pd.DataFrame(export_data)
            st.download_button(
                label="Download Packing Plan as CSV",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name='truck_packing_plan.csv',
                mime='text/csv'
            )

            excel_buffer = pd.ExcelWriter("packing_plan.xlsx", engine='xlsxwriter')
            export_df.to_excel(excel_buffer, index=False, sheet_name='PackingPlan')

            # Add separate truck sheets
            for i, truck in enumerate(trucks):
                pd.DataFrame(truck).to_excel(excel_buffer, index=False, sheet_name=f'Truck {i+1}')

            # Create summary of unique beds
            bed_summary_df = pd.DataFrame(beds)
            unique_beds = bed_summary_df.groupby(['Length', 'Height', 'Width']).size().reset_index(name='Quantity')
            unique_beds.to_excel(excel_buffer, index=False, sheet_name='Bed Summary')

            excel_buffer.close()

            with open("packing_plan.xlsx", "rb") as f:
                st.download_button(
                label="Download Packing Plan as Excel",
                data=f.read(),
                file_name="truck_packing_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("Packing plan generated with separate sheets for trucks and a bed summary.")
                    label="Download Packing Plan as Excel",
                    data=f.read(),
                    file_name="truck_packing_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error computing transport plan: {e}")
