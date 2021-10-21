# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import work_object_detection_pb2 as work__object__detection__pb2


class WorkObjectDetectionStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Predict = channel.unary_unary(
                '/workobjectdetection.WorkObjectDetection/Predict',
                request_serializer=work__object__detection__pb2.Image.SerializeToString,
                response_deserializer=work__object__detection__pb2.Detection.FromString,
                )
        self.SplitAndPredict = channel.unary_unary(
                '/workobjectdetection.WorkObjectDetection/SplitAndPredict',
                request_serializer=work__object__detection__pb2.Image.SerializeToString,
                response_deserializer=work__object__detection__pb2.Detections.FromString,
                )


class WorkObjectDetectionServicer(object):
    """Missing associated documentation comment in .proto file"""

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SplitAndPredict(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_WorkObjectDetectionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=work__object__detection__pb2.Image.FromString,
                    response_serializer=work__object__detection__pb2.Detection.SerializeToString,
            ),
            'SplitAndPredict': grpc.unary_unary_rpc_method_handler(
                    servicer.SplitAndPredict,
                    request_deserializer=work__object__detection__pb2.Image.FromString,
                    response_serializer=work__object__detection__pb2.Detections.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'workobjectdetection.WorkObjectDetection', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class WorkObjectDetection(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/workobjectdetection.WorkObjectDetection/Predict',
            work__object__detection__pb2.Image.SerializeToString,
            work__object__detection__pb2.Detection.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SplitAndPredict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/workobjectdetection.WorkObjectDetection/SplitAndPredict',
            work__object__detection__pb2.Image.SerializeToString,
            work__object__detection__pb2.Detections.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
