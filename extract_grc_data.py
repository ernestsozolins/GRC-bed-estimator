import streamlit as st
import pandas as pd
import pdfplumber
import re
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go

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
                    "Weight": None
                })
    return pd.DataFrame(data)

def extract_from_excel_or_csv(file):
    try:
        df = pd.read_excel(file)
    except:
        df = pd.read_csv(file)

    df = df.dropna(axis=1, how='all')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    mapping = {
        'Type': df.columns[0] if len(df.columns) > 0 else None,
        'Count': df.columns[1] if len(df.columns) > 1 else None,
        'Weight': df.columns[2] if len(df.columns) > 2 else None,
        'Height': df.columns[3] if len(df.columns) > 3 else None,
        'Width': df.columns[4] if len(df.columns) > 4 else None,
        'Depth': df.columns[5] if len(df.columns) > 5 else None
    }

    st.subheader("Adjust Column Mapping (optional)")
    use_defaults = st.checkbox("Use default column mapping (ignore smart detection)", value=False)

    if use_defaults:
        mapping = {
            'Type': df.columns[0] if len(df.columns) > 0 else None,
            'Count': df.columns[1] if len(df.columns) > 1 else None,
            'Weight': df.columns[2] if len(df.columns) > 2 else None,
            'Height': df.columns[3] if len(df.columns) > 3 else None,
            'Width': df.columns[4] if len(df.columns) > 4 else None,
            'Depth': df.columns[5] if len(df.columns) > 5 else None
        }

    type_col = st.selectbox("Column for Type", df.columns, index=df.columns.get_loc(mapping['Type']) if mapping.get('Type') in df.columns else 0)
    count_col = st.selectbox("Column for Count", df.columns, index=df.columns.get_loc(mapping['Count']) if mapping.get('Count') in df.columns else 0)
    height_col = st.selectbox("Column for Height", df.columns, index=df.columns.get_loc(mapping['Height']) if mapping.get('Height') in df.columns else 0)
    width_col = st.selectbox("Column for Width", df.columns, index=df.columns.get_loc(mapping['Width']) if mapping.get('Width') in df.columns else 0)
    depth_col = st.selectbox("Column for Depth", df.columns, index=df.columns.get_loc(mapping['Depth']) if mapping.get('Depth') in df.columns else 0)
    weight_col = st.selectbox("Column for Weight (optional)", ["None"] + df.columns.tolist(), index=(df.columns.get_loc(mapping['Weight']) + 1) if mapping.get('Weight') and mapping['Weight'] in df.columns else 0)

    selected_cols = [type_col, count_col, height_col, width_col, depth_col]
    new_names = ['Type', 'Count', 'Height', 'Width', 'Depth']
    if weight_col != "None":
        selected_cols.append(weight_col)
        new_names.append('Weight')

    try:
        extracted = df[selected_cols]
        extracted.columns = new_names
        extracted = extracted.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        extracted.replace("", pd.NA, inplace=True)
        extracted = extracted.dropna(how='any')

        header_keywords = ['type', 'tips', 'count', 'qty', 'skaits', 'height', 'augstums', 'width', 'platums', 'garums', 'depth', 'dziļums', 'weight', 'svars']
        extracted = extracted[~extracted.apply(lambda row: sum(any(str(val).lower() == kw for kw in header_keywords) for val in row) >= 3, axis=1)]

        return extracted
    except Exception as e:
        st.error(f"Failed to extract data using selected columns. Error: {e}")
        return pd.DataFrame()

st.subheader("Set Packing Orientation per Panel Type")
panel_types = df['Type'].unique()
orientation_map = {}
for pt in panel_types:
    orientation = st.selectbox(f"Packing Orientation for {pt}", options=["Front Face", "Side"], index=0, key=f"orient_{pt}")
    orientation_map[pt] = orientation


def compute_beds_and_trucks(panels, bed_width=2400, bed_weight_limit=2500, truck_weight_limit=15000, truck_max_length=13620, bed_dead_space_length=0, bed_dead_space_height=0, panel_spacing=0, max_bed_height=9999):
    beds = []
    for panel in panels:
        placed = False
        for bed in beds:
            used_depth = sum(p['Depth'] + panel_spacing for p in bed)
            total_weight = sum(p['Weight'] for p in bed)
            max_height = max(p['Height'] for p in bed + [panel]) + bed_dead_space_height
            if used_depth + panel['Depth'] <= bed_width and total_weight + panel['Weight'] <= bed_weight_limit and max_height <= max_bed_height:
                bed.append(panel)
                placed = True
                break
        if not placed:
            beds.append([panel])

    bed_summaries = []
    for bed in beds:
        bed_length = max(p['Width'] for p in bed) + bed_dead_space_length
        bed_height = max(p['Height'] for p in bed) + bed_dead_space_height
        bed_weight = sum(p['Weight'] for p in bed)
        panel_types = [str(p['Type']).strip() for p in bed if pd.notna(p['Type']) and isinstance(p['Type'], str) and str(p['Type']).strip().lower() not in ("", "nan", "none")]
        bed_summaries.append({
            'Length': bed_length,
            'Height': bed_height,
            'Width': bed_width,
            'Weight': bed_weight,
            'Num Panels': len(bed),
            'Panel Types': panel_types
        })

    trucks = []
    for bed in bed_summaries:
        placed = False
        for truck in trucks:
            used_length = sum(b['Length'] for b in truck)
            total_weight = sum(b['Weight'] for b in truck)
            if used_length + bed['Length'] <= truck_max_length and total_weight + bed['Weight'] <= truck_weight_limit:
                truck.append(bed)
                placed = True
                break
        if not placed:
            trucks.append([bed])

    return bed_summaries, trucks

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        df = extract_from_pdf(uploaded_file)
    else:
        df = extract_from_excel_or_csv(uploaded_file)

    if not df.empty:
        st.success("Data extracted successfully!")

        st.subheader("Row Removal and Update Option")
        delete_rows = st.multiselect("Select row indices to delete from extracted data", df.index.tolist(), default=[])
        if delete_rows:
            df = df.drop(delete_rows).reset_index(drop=True)
            st.info("Selected rows have been removed.")

        st.subheader("Extracted GRC Panel Data")
        st.dataframe(df)

        if 'Count' in df.columns:
            df['Count'] = pd.to_numeric(df['Count'], errors='coerce')
            total = df['Count'].sum()
            st.markdown(f"**Total Panel Count:** {total}")

        st.subheader("Visualize a Specific Panel in 3D")
        material_thickness = st.number_input("Base Material Thickness (mm)", value=16)
        
        panel_index = st.selectbox("Select Panel Index to Visualize", options=df.index.tolist(), format_func=lambda i: f"{i}: {df.iloc[i]['Type']}")
        selected_panel = df.loc[panel_index]
        try:
            h, w, d = float(selected_panel['Height']), float(selected_panel['Width']), float(selected_panel['Depth'])
            base_thickness = material_thickness
            total_depth = base_thickness + d

            # Rebuild full panel with front face and solid side reveals
            front_plate = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[0, 0, material_thickness, material_thickness, 0, 0, material_thickness, material_thickness],
                z=[0, 0, 0, 0, h, h, h, h],
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightblue',
                name='Front Plate'
            )

            # Top reveal (along width)
            top_reveal = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[material_thickness, material_thickness, material_thickness + d, material_thickness + d]*2,
                z=[h - material_thickness]*4 + [h]*4,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='orange',
                name='Top Reveal'
            )

            # Bottom reveal (along width)
            bottom_reveal = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[material_thickness, material_thickness, material_thickness + d, material_thickness + d]*2,
                z=[0]*4 + [material_thickness]*4,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='orange',
                name='Bottom Reveal'
            )

            # Left reveal (along height)
            left_reveal = go.Mesh3d(
                x=[0]*4 + [material_thickness]*4,
                y=[material_thickness, material_thickness + d]*2*2,
                z=[0, 0, h, h]*2,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='orange',
                name='Left Reveal'
            )

            # Right reveal (along height)
            right_reveal = go.Mesh3d(
                x=[w - material_thickness]*4 + [w]*4,
                y=[material_thickness, material_thickness + d]*2*2,
                z=[0, 0, h, h]*2,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='orange',
                name='Right Reveal'
            )

            rear_plate = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[material_thickness + d] * 4 + [material_thickness + d] * 4,
                z=[0, 0, 0, 0, h, h, h, h],
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightblue',
                name='Rear Wall'
            )

            fill_top = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[0, 0, material_thickness + d, material_thickness + d] * 2,
                z=[h] * 4 + [h] * 4,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightgray',
                name='Top Fill'
            )

            fill_bottom = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[0, 0, material_thickness + d, material_thickness + d] * 2,
                z=[0] * 4 + [0] * 4,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightgray',
                name='Bottom Fill'
            )

            fill_left = go.Mesh3d(
                x=[0] * 4 + [0] * 4,
                y=[0, 0, material_thickness + d, material_thickness + d] * 2,
                z=[0, h, h, 0] * 2,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightgray',
                name='Left Fill'
            )

            fill_right = go.Mesh3d(
                x=[w] * 4 + [w] * 4,
                y=[0, 0, material_thickness + d, material_thickness + d] * 2,
                z=[0, h, h, 0] * 2,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightgray',
                name='Right Fill'
            )

            fill_core = go.Mesh3d(
                x=[0, w, w, 0, 0, w, w, 0],
                y=[0] * 4 + [material_thickness] * 4,
                z=[0, 0, h, h] * 2,
                i=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                j=[1, 2, 3, 3, 0, 1, 5, 6, 7, 7, 4, 5],
                k=[2, 3, 0, 1, 2, 3, 6, 7, 4, 5, 6, 7],
                opacity=1,
                color='lightgray',
                name='Core Fill'
            )

            shapes = [front_plate, rear_plate, top_reveal, bottom_reveal, left_reveal, right_reveal, fill_core,
                      fill_top, fill_bottom, fill_left, fill_right]
            

                            

            fig = go.Figure(data=shapes)
            # Annotate panel dimensions
            fig.add_trace(go.Scatter3d(
                x=[w / 2], y=[-material_thickness * 2], z=[0],
                mode="text",
                text=[f"Width: {w} mm"],
                textposition="bottom center",
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[-material_thickness * 2], y=[material_thickness / 2], z=[h / 2],
                mode="text",
                text=[f"Height: {h} mm"],
                textposition="middle right",
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[w + material_thickness], y=[material_thickness + d / 2], z=[h + material_thickness],
                mode="text",
                text=[f"Depth: {d} mm"],
                textposition="top center",
                showlegend=False
            ))
            fig.update_layout(
                title=f"3D Visualization of Panel {panel_index} ({selected_panel['Type']})",
                scene=dict(
                    xaxis_title='Width (mm)',
                    yaxis_title='Depth (mm)',
                    zaxis_title='Height (mm)'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Could not generate 3D panel view: {e}")

        st.subheader("Packing Configuration")
        bed_width = st.number_input("Bed Width (mm)", value=2400)
        bed_weight_limit = st.number_input("Bed Weight Limit (kg)", value=2500)
        truck_weight_limit = st.number_input("Truck Weight Limit (kg)", value=15000)
        truck_max_length = st.number_input("Truck Max Length (mm)", value=13620)
        panel_thickness = st.number_input("Panel Thickness (mm, if no weight provided)", value=30)
        bed_dead_space_length = st.number_input("Dead Space in Bed Length Direction (mm)", value=0)
        bed_dead_space_height = st.number_input("Dead Space in Bed Height Direction (mm)", value=0)
        panel_spacing = st.number_input("Optional Spacing Between Panels (mm)", value=0)
        max_bed_height = st.number_input("Maximum Bed Height (mm)", value=9999)
        
        density = 2100

        if st.button("Run Transport Analysis"):
            try:
                panel_rows = []
                for _, row in df.iterrows():
                    count = int(row['Count']) if 'Count' in row and pd.notna(row['Count']) else 1
                    for _ in range(count):
                        try:
                            height = float(row['Height'])
                            width = float(row['Width'])
                            depth = float(row['Depth'])
                            panel_type = row['Type']
                            orientation = orientation_map.get(panel_type, "Front Face")
                            if orientation == "Side":
                              height, depth = depth, height
                            weight = float(row['Weight']) if 'Weight' in row and pd.notna(row['Weight']) else ((2 * height * width + 2 * width * depth + 2 * height * depth) * panel_thickness / 1e9) * density
                            panel_rows.append({
                                'Type': row['Type'],
                                'Height': height,
                                'Width': width,
                                'Depth': depth,
                                'Weight': weight
                            })
                        except Exception as ex:
                            st.warning(f"Skipping row due to invalid data: {ex}")

                beds, trucks = compute_beds_and_trucks(panel_rows, bed_width, bed_weight_limit, truck_weight_limit, truck_max_length, bed_dead_space_length, bed_dead_space_height, panel_spacing, max_bed_height)

                st.markdown(f"**Total Beds:** {len(beds)}")
                st.markdown(f"**Total Trucks:** {len(trucks)}")

                usage_data = pd.DataFrame([b for b in beds if b['Height'] <= max_bed_height])

                fig, ax = plt.subplots()
                ax.bar(range(len(usage_data)), usage_data['Length'])
                ax.set_title("Used Bed Length per Bed")
                ax.set_xlabel("Bed Index")
                ax.set_ylabel("Length (mm)")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.bar(range(len(usage_data)), usage_data['Height'])
                ax.set_title("Used Bed Height per Bed")
                ax.set_xlabel("Bed Index")
                ax.set_ylabel("Height (mm)")
                st.pyplot(fig)

                truck_weights = [sum(b['Weight'] for b in truck) for truck in trucks]
                truck_lengths = [sum(b['Length'] for b in truck) for truck in trucks]

                fig, ax = plt.subplots()
                ax.bar(range(len(trucks)), truck_weights)
                ax.set_title("Total Weight per Truck")
                ax.set_xlabel("Truck Index")
                ax.set_ylabel("Weight (kg)")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.bar(range(len(trucks)), truck_lengths)
                ax.set_title("Used Length per Truck")
                ax.set_xlabel("Truck Index")
                ax.set_ylabel("Length (mm)")
                st.pyplot(fig)

                for i, truck in enumerate(trucks):
                    st.markdown(f"### Truck {i+1} - Beds: {len(truck)}")
                    st.dataframe(pd.DataFrame(truck))

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
                st.download_button("Download Packing Plan as CSV", data=export_df.to_csv(index=False).encode('utf-8'), file_name='truck_packing_plan.csv', mime='text/csv')

                with io.BytesIO() as buffer:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='PackingPlan')
                        for i, truck in enumerate(trucks):
                            pd.DataFrame(truck).to_excel(writer, index=False, sheet_name=f'Truck {i+1}')

                        bed_summary_df = pd.DataFrame(beds)
                        unique_beds = bed_summary_df.groupby(['Length', 'Height', 'Width']).size().reset_index(name='Quantity')
                        unique_beds.to_excel(writer, index=False, sheet_name='Bed Summary')

                    st.download_button("Download Packing Plan as Excel", data=buffer.getvalue(), file_name="truck_packing_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.success("Packing plan generated with separate sheets for trucks and a bed summary.")

                st.subheader("Mass Summary by Panel Type")
                panel_df = pd.DataFrame(panel_rows)
                panel_summary = panel_df.groupby('Type').agg(
                    Total_Quantity=('Type', 'count'),
                    Total_Weight_kg=('Weight', 'sum'),
                    Average_Weight_kg=('Weight', 'mean'),
                    Height_mm=('Height', 'first'),
                    Width_mm=('Width', 'first'),
                    Depth_mm=('Depth', 'first')
                ).reset_index()
                st.dataframe(panel_summary)
                st.download_button("Download Mass Summary as CSV", data=panel_summary.to_csv(index=False).encode('utf-8'), file_name='panel_mass_summary.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Error computing transport plan: {e}")
