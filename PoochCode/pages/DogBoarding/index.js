import React from "react";
import BoardingForm from "../../components/BoardingForm"; // Inner Component
import Fade from "react-reveal"; // Animation

class DogBoarding extends React.Component {
  render() {
    return (
        <Fade> <BoardingForm /> </Fade>
    );
  }
}

export default DogBoarding;