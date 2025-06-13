import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPolygon, Polygon, Point
from pyproj import Transformer
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import alphashape

st.set_page_config(layout="wide", page_title="Field Boundary Detection")

st.title("Field Boundary Detection from GPS Data")

uploaded_file = st.file_uploader("Upload CSV with 'timestamp', 'latitude', 'longitude'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    required_cols = {'timestamp', 'latitude', 'longitude'}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        st.stop()

    df = df[['timestamp', 'latitude', 'longitude']].dropna().drop_duplicates().reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    st.write(f"Loaded {len(df)} GPS points after cleaning.")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:24378", always_xy=True)
    df['x'], df['y'] = transformer.transform(df['longitude'].values, df['latitude'].values)
    coords = df[['x', 'y']].values

    if len(coords) < 10:
        st.warning("Not enough data points for clustering (min 10 required).")
        st.stop()

    # DBSCAN clustering
    db = DBSCAN(eps=3.1, min_samples=18, metric='euclidean')
    df['cluster'] = db.fit_predict(coords)
    df['label'] = df['cluster'].apply(lambda x: 'road' if x == -1 else 'field')

    # Post-processing: Correct road points inside field clusters
    tree = BallTree(coords, metric='euclidean')
    radius = 10
    indices = tree.query_radius(coords, r=radius)

    def corrected_label(i):
        neighbors = indices[i]
        field_count = (df.iloc[neighbors]['label'] == 'field').sum()
        road_count = len(neighbors) - field_count
        return 'field' if df.at[i, 'label'] == 'road' and field_count > road_count else df.at[i, 'label']

    df['label'] = [corrected_label(i) for i in range(len(df))]

    st.write("Clustering and relabeling complete.")

    # Create alpha shapes
    union_hull = None
    alpha = 0.08
    hulls_info = []

    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_df = df[(df['cluster'] == cluster_id) & (df['label'] == 'field')]
        if len(cluster_df) < 4:
            continue
        try:
            hull = alphashape.alphashape(list(zip(cluster_df['x'], cluster_df['y'])), alpha)
            if hull.geom_type == 'Polygon':
                union_hull = hull if union_hull is None else union_hull.union(hull)
        except:
            continue

    if union_hull is not None:
        # Remove field points outside union hull
        def is_inside(x, y):
            pt = Point(x, y)
            if isinstance(union_hull, MultiPolygon):
                return any(poly.contains(pt) for poly in union_hull.geoms)
            return union_hull.contains(pt)

        df['label'] = [label if label == 'road' or is_inside(x, y) else 'road'
                       for label, x, y in zip(df['label'], df['x'], df['y'])]

        st.write("Field points outside boundary removed.")

    # Map visualization
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=17, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=True, control=True
    ).add_to(m)

    folium.LayerControl().add_to(m)

    for _, row in df.iterrows():
        color = 'blue' if row['label'] == 'field' else 'red'
        folium.CircleMarker([row['latitude'], row['longitude']], radius=2, color=color,
                            fill=True, fill_opacity=0.6).add_to(m)

    transformer_inv = Transformer.from_crs("EPSG:24378", "EPSG:4326", always_xy=True)
    hull_count = 0

    if union_hull:
        polygons = union_hull.geoms if isinstance(union_hull, MultiPolygon) else [union_hull]
        for polygon in polygons:
            hull_count += 1
            latlon_coords = [transformer_inv.transform(x, y)[::-1] for x, y in polygon.exterior.coords]
            area_gunthas = polygon.area / 101.171367
            folium.Polygon(locations=latlon_coords, color='green', fill=True, fill_opacity=0.3,
                           popup=f"Area: {area_gunthas:.2f} gunthas",
                           tooltip=f"Area: {area_gunthas:.2f} gunthas").add_to(m)

            centroid = transformer_inv.transform(*polygon.centroid.coords[0])[::-1]
            folium.Marker(centroid, popup=f"Hull_{hull_count}: {area_gunthas:.2f} gunthas").add_to(m)

            hulls_info.append({"Hull": f"Hull_{hull_count}", "Area (gunthas)": round(area_gunthas, 2)})

    st.subheader("Field Map")
    st_data = st_folium(m, width=900, height=600)

    st.subheader("Detected Field Areas (Gunthas)")
    if hulls_info:
        df_hulls = pd.DataFrame(hulls_info).sort_values("Area (gunthas)", ascending=False).reset_index(drop=True)
        st.dataframe(df_hulls, use_container_width=True)
    else:
        st.info("No valid field hulls detected.")

else:
    st.info("Please upload a CSV file to begin.")
