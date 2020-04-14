import React from "react";
import LoginForm from "../../components/LoginForm"; // Inner component
import Fade from "react-reveal"; // Animation

class Login extends React.Component {
  render() {
    return (
    <Fade> <LoginForm/> </Fade>
    );
  }
}

export default Login;