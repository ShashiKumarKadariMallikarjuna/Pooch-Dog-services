import React from "react";
import MeetupForm from "../../components/MeetupForm"; // Inner component
import Fade from "react-reveal"; // Animation

class DogMeetup extends React.Component {
  render() {
    return (
        <Fade> <MeetupForm /> </Fade>
    );
  }
}

export default DogMeetup;