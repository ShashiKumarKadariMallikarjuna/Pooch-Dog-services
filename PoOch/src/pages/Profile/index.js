import React from "react";
import ProfileForm from "../../components/ProfileForm"; // Inner component
import Fade from "react-reveal"; // Animation

class Profile extends React.Component {
  render() {
    return (
    <Fade> <ProfileForm/> </Fade>
    );
  }
}

export default Profile;
