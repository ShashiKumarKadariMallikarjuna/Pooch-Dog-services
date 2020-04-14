import React from "react";
import WalkerForm from "../../components/WalkerForm"; // Inner Component
import Fade from "react-reveal"; // Animation

class DogWalking extends React.Component {
  render() {
    return (
        <Fade> <WalkerForm /> </Fade>
    );
  }
}

export default DogWalking;