import os
class CheckpointModel():
    def __init__(self, save_config):
      """
        save_config : dict
            type : ["best_model_loss", "every_tot", "first_train", "last_train"]
            times : [OPT - if type is "every_tot"] int, number of epoch when save
            overwrite : boolean
            path_file : path where save
            name_file : name of file
            path_not_exist_mode: if path not exist: "create","except", default:except
            path_exist_mode: "use","clean","new" default:"use"
      """
      self.type_checkpointer = ["best_model_loss", "every_tot", "first_train", "last_train"]
      self.checher = dict()
      if save_config == None:
          self.checher['enable'] = False
      else:
          self.checher['enable'] = True
          is_types_safe = True
          not_types_safe = list()

          for type_config in save_config['type']:
            if type_config not in self.type_checkpointer:
              is_types_safe = False
              not_types_safe.append(type_config)

          if is_types_safe:
              self.checher["type"] = save_config['type']

              dirpath_save = save_config["path_file"]
              self.checher["name_file"] = save_config["name_file"]

              if "path_not_exist" in save_config:
                  path_not_exist_mode = save_config["path_not_exist"]
              else:
                  path_not_exist_mode = "except"

              if "path_exist" in save_config:
                  path_exist_mode = save_config["path_exist"]
              else:
                  path_exist_mode = "use"


              if os.path.isdir(dirpath_save): #path esiste
                  if path_exist_mode == "use":
                      self.checher["path_file"] = dirpath_save
                  elif path_exist_mode == "clean":
                      self.checher["path_file"] = Util_class.folder_manage(dirpath_save, clean=True)
                  elif path_exist_mode == "new":
                      self.checher["path_file"] = Util_class.folder_manage(dirpath_save, uniquify=True)
                  else:
                      raise CheckpointModel_Exception_ParamPathNotRecoignezed("Exist", path_exist_mode)

              else: #path non esiste
                  if path_not_exist_mode == "create":
                      self.checher["path_file"] = Util_class.folder_manage(dirpath_save, uniquify=True)
                  elif path_not_exist_mode == "except":
                      raise CheckpointModel_Exception_SavePathNotExist(dirpath_save)
                  else:
                      raise CheckpointModel_Exception_ParamPathNotRecoignezed("NotExist", path_exist_mode)

              print("Your model's checkpoint is save in : {fpath}".format(fpath = self.checher["path_file"]))

              if "overwrite" in save_config:
                  self.checher["overwrite"] = save_config["overwrite"]
              else:
                  self.checher["overwrite"] = False

              for type_config in save_config['type']:
                  if type_config == "every_tot":
                      self.checher["times"] = save_config['times']
                      self.checher["next_epoch"] = save_config['times']
                  elif type_config == "best_model_loss":
                      self.checher["last_loss"] = None
          else:
              raise CheckpointModel_Exception_TypeChecker(not_types_safe)

    def checkToSave(self, graphe_model, epoch, epochs, loss, phase=None):
        from graph_e_model_mlp import GraphEModel as MLPModel
        from graph_e_model_gcn import GraphEModel as GCNModel
        from graph_e_model_gcn_smooth import GraphEModel as GCNSmoothModel
        if not isinstance(graphe_model, (MLPModel, GCNModel, GCNSmoothModel)):
            raise CheckpointModel_Exception_GraphEModelType(graphe_model)
        else:
          to_save = False

          for type_config in self.checher['type']:
              if type_config == "every_tot":
                  if self.checher['next_epoch'] == epoch:
                      to_save = True
                      self.checher['next_epoch'] += self.checher["times"]
              elif type_config == "best_model_lost":
                  if self.checher['last_loss'] > loss:
                      to_save = True
                      self.checher['last_loss'] = loss
              elif type_config == "first_train":
                  if epoch==1:
                    to_save = True
              elif type_config == "last_train":
                  if epoch == epochs:
                    to_save = True

          if phase is not None:
              _phase = "_phase" + phase
          else:
              _phase = ""
          if to_save:
              if self.checher["overwrite"]:
                  path_chechpoint_file = "{fpath}/{fname}{fphase}.carbo".format(fpath = self.checher["path_file"], fname = self.checher["name_file"], fphase=_phase)
              else:
                  for type_config in self.checher['type']:
                      if type_config in ["every_tot", "first_train", "last_train"]:
                          path_chechpoint_file = "{fpath}/{fname}{fphase}_epoch_{fepoch}.carbo".format(fpath = self.checher["path_file"], fname = self.checher["name_file"], fepoch = epoch, fphase=_phase)
                      elif type_config in ["best_model_lost"]:
                          path_chechpoint_file = "{fphase}_epoch_{fepoch}_loss_{floss:.8f}.carbo".format(fpath = self.checher["path_file"],fname = self.checher["name_file"], fepoch = epoch, floss = loss, fphase=_phase)
              graphe_model.save_model(epoch = epoch, path_file = path_chechpoint_file)
              print("Epoch : ",epoch,"/",epochs,"\tLoss : ", loss, "\tmodel checkpoint saved as: {fpath}".format(fpath=path_chechpoint_file))


class CheckpointModel_Exception_TypeChecker(Exception):
      """Exception raised for errors in activation function type"""

      def __init__(self, value):
          self.value = value

      def __str__(self):
          return f'{self.value} : type of checkpointer not recognized.'

class CheckpointModel_Exception_GraphEModelType(Exception):
      """Exception raised for errors in activation function type"""

      def __init__(self, value):
          self.value = type(value)

      def __str__(self):
          return f'Model should be a GraphEModel object but checker receiver a {self.value} type object.'

class CheckpointModel_Exception_SavePathNotExist(Exception):
      """Exception raised for errors in activation function type"""

      def __init__(self, path):
          self.path = path

      def __str__(self):
          return f"Your model's checkpoint could be save because '{self.path}' not exist."

class CheckpointModel_Exception_ParamPathNotRecoignezed(Exception):
      """Exception raised for errors of path to save embedding is none"""

      def __init__(self,mode,value):
          self.value = value
          self.mode = mode

      def __str__(self):
          return f"{self.mode} modality param {self.value} is recognized."