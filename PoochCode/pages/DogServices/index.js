import React from "react";
import ServicesForm from "../../components/ServicesForm"; // Inner component
import Fade from "react-reveal"; // Animation

class DogServices extends React.Component {
  render() {
    return (
        <Fade> <ServicesForm/> </Fade>
    );
  }
}

export default DogServices;