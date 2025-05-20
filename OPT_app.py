import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint, Polygon, Point, MultiPolygon
from pyproj import Transformer
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numpy as np
from shapely.ops import unary_union
from concave_hull import concave_hull
import alphashape

st.set_page_config(page_title="Field Detection & Area Analysis", layout="wide")
st.title("Field Detection from GPS Data")

uploaded_file = st.file_uploader("Upload GPS CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[['timestamp', 'latitude', 'longitude']].dropna().reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.drop_duplicates(subset=['latitude', 'longitude'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:24378", always_xy=True)
    df['x'], df['y'] = transformer.transform(df['longitude'].values, df['latitude'].values)
    coords = df[['x', 'y']].values

    eps_m = 3.10
    min_samples = 18
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean')
    df['cluster'] = db.fit_predict(coords)

    df['label'] = df['cluster'].apply(lambda x: 'road' if x == -1 else 'field')
    tree = BallTree(coords, metric='euclidean')
    radius = 10
    indices = tree.query_radius(coords, r=radius)

    labels_corrected = []
    for i, neighbors in enumerate(indices):
        field_count = sum(df.iloc[neighbors]['label'] == 'field')
        road_count = sum(df.iloc[neighbors]['label'] == 'road')
        if df.iloc[i]['label'] == 'road' and field_count > road_count:
            labels_corrected.append('field')
        else:
            labels_corrected.append(df.iloc[i]['label'])
    df['label'] = labels_corrected

    union_hull = None
    alpha = 0.08
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_df = df[(df['cluster'] == cluster_id) & (df['label'] == 'field')]
        if len(cluster_df) < 150:
            continue

        points_itm = list(zip(cluster_df['x'], cluster_df['y']))

        try:
            hull = alphashape.alphashape(points_itm, alpha)
            if hull.geom_type == 'Polygon':
                if union_hull is None:
                    union_hull = hull
                else:
                    union_hull = union_hull.union(hull)
        except Exception as e:
            st.warning(f"Skipping cluster {cluster_id} due to concave hull error: {e}")

    for i, row in df[df['label'] == 'field'].iterrows():
        point = Point(row['x'], row['y'])
        inside = False

        if isinstance(union_hull, MultiPolygon):
            for polygon in union_hull.geoms:
                if polygon.contains(point):
                    inside = True
                    break
        elif union_hull is not None and union_hull.contains(point):
            inside = True

        if not inside:
            df.at[i, 'label'] = 'road'

    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=17, max_zoom=20, control_scale=True)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=True,
        control=True
    ).add_to(m)

    for _, row in df.iterrows():
        color = "blue" if row['label'] == 'field' else "red"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    centroids = {}
    hull_count = 0
    if union_hull is not None:
        hulls = union_hull.geoms if isinstance(union_hull, MultiPolygon) else [union_hull]

        for i, polygon in enumerate(hulls):
            hull_latlon = [transformer.transform(x, y, direction='INVERSE')[::-1] for x, y in polygon.exterior.coords]
            area_m2 = polygon.area
            area_gunthas = area_m2 / 101.171367

            centroid = polygon.centroid
            centroid_latlon = [transformer.transform(centroid.x, centroid.y, direction='INVERSE')[::-1]]

            hull_count += 1
            centroid_name = f"Hull_{hull_count}"

            centroids[centroid_name] = {
                'centroid': centroid_latlon,
                'area': area_gunthas
            }

            folium.Polygon(
                locations=hull_latlon,
                color='green',
                fill=True,
                fill_opacity=0.3,
                popup=f"{centroid_name} - Area: {area_gunthas:.2f} gunthas"
            ).add_to(m)

            folium.Marker(
                location=centroid_latlon[0],
                popup=f"{centroid_name} - Area: {area_gunthas:.2f} gunthas"
            ).add_to(m)

    st.subheader("Map Output")
    st_folium(m, width=1000, height=600)

    st.subheader("Hull Area Summary")
    area_data = [{'Hull': k, 'Area (gunthas)': v['area']} for k, v in centroids.items()]
    st.table(pd.DataFrame(area_data))
