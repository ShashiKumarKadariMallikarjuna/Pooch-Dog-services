import React from "react";
import PetForm from "../../components/PetForm"; // Inner Component
import Fade from "react-reveal"; // Animation

class YourPets extends React.Component {
  render() {
    return (
      <Fade> <PetForm /> </Fade>
    );
  }
}

export default YourPets;