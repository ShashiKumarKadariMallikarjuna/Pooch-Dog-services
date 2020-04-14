import React from "react";

//Graphics, Styles, Animation
import "./styles.scss";
import Fade from "react-reveal";
import Slide from "react-reveal";
import FamilyPets from './media/familypets.jpg';
import DogWalking from './media/dogwalking.jpg';
import DogBoarding from './media/dogboarding.jpg';
import DogGrooming from './media/grooming.jpg';
import DogPark from './media/dogpark.jpg';
import Vet from './media/vet.jpg';

// Components
import Slideshow from "../../components/Slideshow";
import Signup from "../Signup";
import Login from "../Login";
import ChangePassword from "../ChangePassword";

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
    };

    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  showSignup() {
    document.getElementById("signup-box").className="fixed-top collapse.show";
    document.getElementById("signup-box").style.width = "33%";
    document.getElementById("signup-box").style.marginLeft = "33%";
  }

  render() {
    return (
      <div id="home-page">
        <div className="fixed-top home-box">
          <Login/>
        </div>
        <div className="fixed-top home-box">
          <Signup/>
        </div>
        <div className="fixed-top home-box">
          <ChangePassword/>
        </div>
        
        <div id="section1"/>
        <Slideshow/> 
        <div className="row mb-5 mt-3">
          <Slide left>
            <div className="row pt-6 home-row" id="section2">
              <div className="col mx-6 paragraph">
                <font color = "#FFC375"> <center><div className="mb-2 paragraph-title">YOUR PETS</div></center> </font>
                Add pets to your profile to let your future vet, dog walker, or dog groomer get to know your pooch before
                meeting them. Keep track of your pets and show them off to the world. With Pooch, your pets have never
                been closer to everything they need to stay happy and healthy.
              </div>
              <div className="col mx-6">
                <img className="paragraph-pic" src={FamilyPets} alt="Family pets"/>
              </div>
            </div>
            </Slide>
            <Slide right>
            <div className="row home-row py-2" id="section3">
              <div className="col mx-6">
                <img className="paragraph-pic" src={DogWalking} alt="Walking dogs"/>
              </div>
              <div className="col mx-6 paragraph">
              <font color = "#f5dc6b"> <center><div className="mb-2 paragraph-title">DOG WALKING</div></center> </font>
                Find a dog walker that matches your desired experience level, price, and availability. With Pooch, you
                can find verified walkers with reviews from your friends and neighbors. View their profiles and schedule
                a walk that works when you do. Sign up to be a dog walker if you have a passion for pups.
                </div>
            </div>
            </Slide>
            <Slide left>
            <div className="row home-row py-2" id="section4">
              <div className="col mx-6 paragraph">
              <font color = "#9ef566"> <center><div className="mb-2 paragraph-title">DOG BOARDING</div></center> </font>
                Find the most qualified local dog boarders for your four-legged friend when you go out of town. Whether
                you need a provider with no appointment needed, spacious rooms, or even a pool, Pooch has you covered.
                Let your dog enjoy their stay to the fullest.
              </div>
              <div className="col mx-6">
                <img className="paragraph-pic" src={DogBoarding} alt="Dog boarding"/>
              </div>
            </div>
            </Slide>
            <Slide right>
            <div className="row home-row py-2" id="section5">
              <div className="col mx-6">
                <img className="paragraph-pic" src={DogGrooming} alt="Dog grooming"/>
              </div>
              <div className="col mx-6 paragraph">
              <font color = "#6bf5a5"> <center><div className="mb-2 paragraph-title">DOG SERVICES</div></center> </font>
                Find vets, pet supply stores, dog groomers, and adoption centers all without leaving Pooch. With your 
                dog's info easily accessible, speed up the process of finding the dog services you need. Take the care
                of your pooch to the next level with highly-reccomended dog service providers at your fingertips.
              </div>
            </div>
            </Slide>
            <Slide left>
            <div className="row home-row py-2" id="section6">
              <div className="col mx-6 paragraph"> 
              <font color = "#6bd5f5"> <center><div className="mb-2 paragraph-title">DOG MEETUP</div></center> </font> 
              Meetups help people come together for doggy dates without the need to hand out your phone number. With Pooch,
              there's no more guessing when the dog park is full of friends. Let your pooch enjoy time with other dogs 
              in a safe community being surrounded by magnificent dog lovers. Don’t wait for the event, make your own. 
              </div>
              <div className="col mx-6">
                <img className="paragraph-pic" src={DogPark} alt="Dog park"/>
              </div>
            </div>
            </Slide>
            <Slide right>
            <div className="row home-row py-2" id="section7">
              <div className="col mx-6">
                <img className="paragraph-pic" src={Vet} alt="Vet visit"/>
              </div>
              <div className="col mx-6 paragraph">
              <font color = "#BC75FF"> <center><div className="mb-2 paragraph-title">DOG RECORDS</div></center> </font> 
              Keeping track of all the veterinary records for your dog is important and we know how easy 
              it can be to lose a document that may determine what type of care and treatment your pet may need. 
              With Pooch, there's no wasting time looking for records. Upload your pets documents and other essentials. 
              Pooch will have them handy when you need them. Go paperless with Pooch!
              </div>
            </div>
            </Slide>
        </div>
      </div>
    );
  };
}

export default Home;
