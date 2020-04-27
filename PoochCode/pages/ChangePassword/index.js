import React from "react";
import PasswordForm from "../../components/PasswordForm"; // Inner component
import Fade from "react-reveal"; // Animation

class ChangePassword extends React.Component {
  render() {
    return (
          <Fade> <PasswordForm/> </Fade>
    );
  }
}

export default ChangePassword;
