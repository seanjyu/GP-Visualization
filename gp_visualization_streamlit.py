
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from gp_implementation import gp_implementation

class streamlit_gp_visualization:

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def create_canvas(self):
        st.title('Gaussian Process Visualization')
        st.markdown(
            'The following webapp provides a visualization for gaussian'
            'processes. Enter the kernel parameters on the left side bar '
            'and click in the box below to enter data.'
            ' The gaussian process was implemented using only numpy'
            'and the code can be found on the following github repo'
            ': https://github.com/seanjyu/GP-Visualization.')

        st.sidebar.title("RBF Kernel Parameters")
        st.sidebar.number_input("Variance: ", key="variance",
                                min_value=0.01, step=0.01
                                )
        st.sidebar.number_input("Length Scale: ", key="length_scale",
                                min_value=0.01, step=0.01
                                )

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#000000",
            background_color="#EEEEEE",  # bg_color,
            background_image=None,
            # Image.open(bg_image) if bg_image else None,
            update_streamlit=True,
            height=150,
            drawing_mode="point",
            point_display_radius=3,
            key="canvas",
            )

        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data[
                                            "objects"])  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            # Code below used to see plot objects
            # st.dataframe(objects)

            if len(objects) != 0:
                str_train = np.array([objects["left"], objects["top"]])
                self.X_train = np.array([objects["left"]]).reshape(-1, 1) / 605
                self.y_train = (-1 * np.array([objects["top"]]).reshape(-1, 1) + 75)/160

                # Calculate and plot gp
                self.calculate_gp()



    def calculate_gp(self):

        gp_class = gp_implementation("rbf",
                                     [st.session_state.variance, st.session_state.length_scale],
                                     1)
        fig, ax = plt.subplots()
        ax.scatter(self.X_train, self.y_train, label="Observations")
        X = np.linspace(start=0, stop=1, num=1000).reshape(-1, 1)
        mean, cov = gp_class.train(self.X_train, X, self.y_train)
        std_dev = np.sqrt(cov.diagonal()).reshape(-1, 1)
        ax.axis('off')

        # Create Plot
        ax.plot(X, mean, label="Mean prediction")
        ax.fill_between(
            X.reshape(-1),
            (mean - 1.96 * std_dev).reshape(-1),
            (mean + 1.96 * std_dev).reshape(-1),
            alpha=0.5,
            label=r"95% confidence interval",
            )
        plt.legend()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='lower left', bbox_to_anchor=(0.2, -0.2))
        st.pyplot(fig)



if __name__ == "__main__":
    streamlit_class = streamlit_gp_visualization()
    streamlit_class.create_canvas()
