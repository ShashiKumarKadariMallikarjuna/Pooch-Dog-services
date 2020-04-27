import React from "react";
import Fire from "../../config/Fire"; // Firebase
import "./styles.scss"; // Styles

class PasswordForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      email: "",
      currentPassword: "",
      newPassword: ""
    };

    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  sendEmail(e) {
    e.preventDefault();
    let email = document.getElementById("change-password-email").value;
    if (email !== "")
    {
      Fire.auth().sendPasswordResetEmail(email).then(function() {
        window.alert("Email sent. Please check your inbox for the reset link");
      }).catch(function(error) {
        window.alert("Error: Password could not be changed");
      });
    }
    else {
      window.alert("Please enter an email first");
    }
  }

  closeChangePassword() {
    document.getElementById("password-box").className = "collapse"
  }

  render() {
    return (
      <div id="password-box" className="collapse">
          <div className="pooch-navbar-item login-title">Recover Password with Email</div>
          <input name="email" id="change-password-email" type="text" className="login-input" placeholder="Email" value={this.state.email} onChange={this.handleChange}/>
          <div className="row login-button-row">
            <div id="email-button" onClick={this.sendEmail}>Send Email</div>
          </div>
          <div id="close-button" onClick={this.closeChangePassword}>Not what you wanted? Click here to close</div>
          <div id="error-message-password"></div>
      </div>
    ); 
  }
};

export default PasswordForm;
