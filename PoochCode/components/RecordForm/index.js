import React from "react";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styling
import Loader from "react-loader-spinner"; // Loader

class RecordForm extends React.Component{

    componentDidMount() {
        document.getElementById("error-message").style.display = "none";
        document.getElementById("loader").style.display = "block";
        document.getElementById("upload-loader").style.display = "none";
        
        setTimeout(() => { // Show all uploaded records
            Fire.firestore().collection("records")
            .where("email", "==", Fire.auth().currentUser.email)
            .get()
            .then(snapshot => {
              snapshot.docs.forEach(doc => {
                var newPic = document.createElement("img");
                newPic.id = "record";
                newPic.alt = "Document";
                newPic.src = doc.data().pic;
                newPic.docID = doc.id;
                //newPic.addEventListener("click", ()=>{toggleExpand(newPic)});
                newPic.addEventListener("click", ()=>{deletePopup(doc.data().pic, doc.id)});
                document.getElementById("record-row").appendChild(newPic);
              });
            });
          }, 1750);
          function deletePopup(pic, id) {
            document.getElementById("delete-record-popup").className = "fixed-top collapse.show";
            document.getElementById("delete-preview").src = pic;
            document.getElementById("delete-preview").docID = id;
            document.getElementsByClassName("page-footer")[0].className = "page-footer font-small trak_body_small text-left pt-4 px-3 py-3 collapse";
            document.getElementById("upload-box").className = "collapse";
            document.getElementById("record-row").className = "collapse";
            document.getElementById("upload-step1").className = "collapse";
            document.getElementById("upload-step2").className = "collapse";
          }

          function toggleExpand(pic) {
            if (pic.style.maxWidth !== "80vw") {
                pic.style.maxWidth = "80vw";
                pic.style.position = "absolute";
                pic.style.width = "80vw";
                pic.style.top = "25vh";
                pic.style.left = "10vw"; 
            }
            else {
                pic.style.maxWidth = "15vw";
                pic.style.position = "relative";
                pic.style.width = "15vw";
                pic.style.top = "0vh";
                pic.style.left = "0vw"; 
            }
            
        }

          setTimeout(() => {
            document.getElementById("loader").style.display = "none";
         }, 1750);
    }

    filePreview = event => {
        var input = event.target; // Get image that changed state of input element
        var reader = new FileReader();
        reader.onload = function() {
        var dataURL = reader.result;
        var output = document.getElementById('preview');
        output.src = dataURL;
        };
        reader.readAsDataURL(input.files[0]); // Show preview of image
    }

    fileUpload() {
        document.getElementById("upload-loader").style.display = "block";
        Fire.firestore().collection("records").add({ // Save image to database
            email: Fire.auth().currentUser.email,
            pic: document.getElementById("preview").src
        }).catch(error => {
            document.getElementById("error-message").style.display = "block";
          });
          setTimeout(() => { // Only reload page if no error
            document.getElementById("upload-loader").style.display = "none";
            if (document.getElementById("error-message").style.display === "none")
                window.location.reload();
        }, 2250);

    }

    deleteRecord() {
        Fire.firestore().collection("records")
        .where('email', '==', Fire.auth().currentUser.email)
        .get().then(function(querySnapshot) {
        querySnapshot.forEach(function(doc) {
            if (doc.id === document.getElementById("delete-preview").docID)
                doc.ref.delete();
        });
        });
        setTimeout(() => {
           window.location.reload();
        }, 1000);
    }

    closePopup() { 
        document.getElementById("delete-record-popup").className = "fixed-top collapse";
        document.getElementsByClassName("page-footer")[0].className = "page-footer font-small trak_body_small text-left pt-4 px-3 py-3 collapse.show";
        document.getElementById("upload-box").className = "collapse.show";
        document.getElementById("record-row").className = "collapse.show";  
        document.getElementById("upload-step1").className = "collapse.show";
        document.getElementById("upload-step2").className = "collapse.show";
    }

    closeError() { document.getElementById("error-message").style.display = "none"; }

    render(){
        return( 
            <div id="record-form">
                <div className="pooch-navbar-item" id="upload-step1"><b>↙</b> First, select the document you want</div>
                <div id="upload-box">
                    <input className="my-2" type="file" id="file-input" onChange={this.filePreview} />
                    <div className="my-2"><img className="record-preview" id="preview"/></div>
                    <button className="my-2" id="file-upload-button" onClick={this.fileUpload}>Upload Dog Record</button>
                    <div id="upload-loader" className="mt-4"><Loader type="TailSpin" color="black" height={75} width={75}/></div>
                </div>
                <div className="pooch-navbar-item" id="upload-step2"><b>↖</b> Then, click upload</div>
                <div id="loader" className="mt-4"><Loader type="TailSpin" color="black" height={75} width={75}/></div>
                <div className="mt-3" id="record-row"></div>

                <div id="delete-record-popup" className="fixed-top collapse">
                    <div className="pooch-navbar-item my-2">What would you like to do with the current document?</div>
                    <div><button className="my-2 mr-5 btn-danger popup-button" onClick={this.deleteRecord}>Delete</button><button className="my-2 ml-5 btn-primary popup-button" onClick={this.closePopup}>Minimize</button></div>
                    <div><img className="record-preview" id="delete-preview"/></div>
                </div>
                <div className="fixed-top" id="error-message">
                    <div>Image size too large</div>
                    <button className="my-2 mr-4 btn-danger popup-button" onClick={this.closeError}>Close</button>
                </div>
            </div>
        );
    }
}

export default RecordForm;