import React from "react";
import SignUpForm from "../../components/SignUpForm"; // Inner Component
import Fade from "react-reveal"; // Animation

class SignUp extends React.Component {
  render() {
    return (
      <Fade> <SignUpForm/> </Fade>
    );
  }
}

export default SignUp;