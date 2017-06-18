#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "bot.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using bot::ActionRequest;
using bot::ActionResponse;
using bot::BotService;

class BotClient {
 public:
  BotClient(std::shared_ptr<Channel> channel)
      : stub_(Greeter::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::vector<string> GetAction(const ActionRequest& request) {

    // Container for the data we expect from the server.
    ActionResponse response;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->SayHello(&context, request, &response);

    // Act upon its status.
    if (status.ok()) {
      std::vector<string> actions;
      for (int i = 0; i < response.actions_size(); i++) {
        actions.push_back(response.actions(i));
      } 
      return actions;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<BotService::Stub> stub_;
};
